# Copyright (c) 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import sys
import os
import os.path

import tensorflow as tf

from .callbacks.early_stopping import EarlyStoppingCallback
from .callbacks.save_model_checkpoint import CustomSaveModelCheckpoint
from .config import Config, most_recent_model
from .input_fn import input_fn
from .model import DreamGoNet, CustomTensorBoardCallback
from .optimizers.schedules.learning_rate_schedule import WarmupExponentialDecaySchedule
from .ffi.libdg_go import get_num_features

def setUp():
    if os.getenv('DG_CHECK_NUMERICS'):
        tf.debugging.enable_check_numerics()
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    for physical_device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(physical_device, True)

def get_latest_checkpoint(checkpoint_dir):
    files = [file for file in os.listdir(checkpoint_dir) if file.endswith('.h5')]
    latest_checkpoint = max(files, key=lambda file: os.path.getmtime(f'{checkpoint_dir}/{file}'))

    if latest_checkpoint is not None:
        return f'{checkpoint_dir}/{latest_checkpoint}'

def main(args=None, *, base_model_dir='models', model_fn=DreamGoNet):
    config = Config(args)

    if config.is_start():
        model_dir = config.get_model_dir(base_model_dir=base_model_dir)
    else:
        model_dir = config.get_model_dir(base_model_dir=base_model_dir, default=most_recent_model())

    learning_rate = WarmupExponentialDecaySchedule(
        initial_learning_rate=config.initial_learning_rate,
        max_learning_rate=config.max_learning_rate,
        num_warmup_steps=config.num_warmup_steps,
        num_decay_steps=config.num_decay_steps,
        decay_rate=config.decay_rate
    )
    model = model_fn(
        num_blocks=config.num_blocks,
        num_dynamics_blocks=config.num_dynamics_blocks,
        num_unrolls=config.num_unrolls,
        num_channels=config.num_channels,
        num_dynamics_channels=config.num_dynamics_channels,
        num_value_channels=config.num_value_channels,
        num_policy_channels=config.num_policy_channels,
        policy_coefficient=config.policy_coefficient,
        value_coefficient=config.value_coefficient,
        ownership_coefficient=config.ownership_coefficient,
        batch_size=config.batch_size // config.num_unrolls,
        discount_factor=config.discount_factor,
        weight_decay=config.weight_decay,
        label_smoothing=config.label_smoothing,
        clipnorm=config.clipnorm,
        learning_rate_schedule=learning_rate,
        lz_weights=config.lz_weights,
        run_eagerly=config.run_eagerly
    )

    if config.has_model() or not config.is_start():
        # this will build all of the necessary variables in the model and optimizer
        _ = model(tf.zeros([1, 19, 19, get_num_features()], tf.float16), training=False)
        _ = model.optimizer.iterations

        # restore the checkpoint
        latest_checkpoint = get_latest_checkpoint(model_dir)
        model.load_weights(latest_checkpoint)

    if config.is_start() or config.is_resume():
        early_stopping = EarlyStoppingCallback(
            monitor='val_loss',
            num_warmup_steps=config.num_es_warmup_steps,
            num_samples=config.num_es_samples,
            max_slope=config.max_es_slope
        )

        if config.warm_start:
            latest_checkpoint = get_latest_checkpoint(config.warm_start)
            model.load_weights(latest_checkpoint, skip_mismatch=True)

        model.fit(
            x=input_fn(
                files=config.files,
                batch_size=config.batch_size // config.num_unrolls,
                num_unrolls=config.num_unrolls,
                is_training=True
            ),
            epochs=config.epochs,
            verbose=0,
            callbacks=[
                CustomTensorBoardCallback(model_dir, hparams=config.hparams, early_stopping=early_stopping, learning_rate=learning_rate),
                CustomSaveModelCheckpoint(f'{model_dir}', monitor='val_loss', save_best_only=True),
                early_stopping
            ],
            validation_data=input_fn(
                files=config.files,
                batch_size=config.batch_size // config.num_unrolls,
                num_unrolls=config.num_unrolls,
                is_training=False
            )
        )
    elif config.is_verify():
        # iterate over the entire dataset and collect the metric, which we will
        # then pretty-print as a JSON object to standard output
        model.assign_average_vars(xs=input_fn(files=config.files, batch_size=max(1, config.batch_size // (8 * config.num_unrolls)), is_training=None))
        results = model.evaluate(
            x=input_fn(
                files=config.files,
                batch_size=config.batch_size // config.num_unrolls,
                num_unrolls=config.num_unrolls,
                is_training=False
            ),
            verbose=0,
            return_dict=True
        )

        print(json.dumps(
            results,
            default=lambda x: float(x) if x != int(x) else int(x),  # handle `Decimal` types
            sort_keys=True,
            separators=(',', ': '),
            indent=4
        ))
    elif config.is_dump():
        model.assign_average_vars(xs=input_fn(files=config.files, batch_size=max(1, config.batch_size // (8 * config.num_unrolls)), num_unrolls=config.num_unrolls, is_training=None))
        model.dump_to(sys.stdout)


if __name__ == '__main__':
    setUp()
    main()
