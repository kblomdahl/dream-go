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
from .optimizers.schedules.epoch_decay_with_warmup import EpochDecayWithWarmup

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

    model_config = config.model_config
    hparams = { **config.hparams, **model_config.hparams }
    learning_rate_schedule = EpochDecayWithWarmup(model_config.decay_rate)
    model = model_fn(
        num_unrolls=model_config.num_unrolls,
        embeddings_size=model_config.embeddings_size,
        num_repr_blocks=model_config.num_repr_blocks,
        num_repr_channels=model_config.num_repr_channels,
        num_trans_layers=model_config.num_trans_layers,
        num_pred_layers=model_config.num_pred_layers,
        policy_coefficient=model_config.policy_coefficient,
        value_coefficient=model_config.value_coefficient,
        target_coefficient=model_config.target_coefficient,
        similarity_coefficient=model_config.similarity_coefficient,
        batch_size=model_config.batch_size // model_config.num_unrolls,
        discount_factor=model_config.discount_factor,
        weight_decay=model_config.weight_decay,
        label_smoothing=model_config.label_smoothing,
        clipnorm=model_config.clipnorm,
        learning_rate=model_config.learning_rate,
        lz_weights=model_config.lz_weights
    )

    if config.has_model() or not config.is_start():
        # this will build all of the necessary variables in the model and optimizer
        _ = model(
            tf.zeros([1, model_config.num_unrolls, 19, 19, model.num_feature_channels()], tf.float16),
            training=False
        )
        _ = model.optimizer.iterations

        # restore the checkpoint
        latest_checkpoint = get_latest_checkpoint(model_dir)
        model.load_weights(latest_checkpoint)

    # ensure we have a `model_config` written to the model directory
    if not os.path.isfile(f'{model_dir}/config.json'):
        os.makedirs(model_dir, exist_ok=True)
        with open(f'{model_dir}/config.json', 'w') as f:
            f.write(str(model_config))

    if config.is_start() or config.is_resume():
        early_stopping = EarlyStoppingCallback(
            monitor='val_loss',
            num_warmup_steps=model_config.num_early_stopping_warmup_steps,
            num_samples=model_config.num_early_stopping_samples,
            max_slope=model_config.max_early_stopping_slope
        )

        if config.warm_start:
            latest_checkpoint = get_latest_checkpoint(config.warm_start)
            model.load_weights(latest_checkpoint, skip_mismatch=True)

        model.fit(
            x=input_fn(
                files=model_config.data,
                batch_size=model_config.batch_size // model_config.num_unrolls,
                num_unrolls=model_config.num_unrolls,
                is_training=True
            ),
            epochs=model_config.epochs,
            verbose=0,
            callbacks=[
                CustomTensorBoardCallback(model_dir, hparams=hparams, early_stopping=early_stopping),
                CustomSaveModelCheckpoint(model_dir, monitor='val_loss', save_best_only=True),
                tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule, verbose=0),
                early_stopping,
            ],
            validation_data=input_fn(
                files=model_config.data,
                batch_size=model_config.batch_size // model_config.num_unrolls,
                num_unrolls=model_config.num_unrolls,
                is_training=False
            )
        )
    elif config.is_verify():
        # iterate over the entire dataset and collect the metric, which we will
        # then pretty-print as a JSON object to standard output
        results = model.evaluate(
            x=input_fn(
                files=model_config.data,
                batch_size=model_config.batch_size // model_config.num_unrolls,
                num_unrolls=model_config.num_unrolls,
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
        model.dump_to(sys.stdout)


if __name__ == '__main__':
    setUp()
    main()
