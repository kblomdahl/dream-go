# Copyright (c) 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

from time import time
import json

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp

from .layers import NUM_FEATURES
from .layers.batch_norm import XavierOrthogonalInitializer
from .layers.leela_zero import leela_zero
from .layers.dynamics import Dynamics
from .layers.to_dict import tensor_to_dict
from .layers.features_to_repr import FeaturesToRepr
from .layers.predictions import Predictions
from .layers.rnn import RNN
from .layers.quantize import Quantize
from .optimizers.schedules.learning_rate_schedule import WarmupExponentialDecaySchedule

class DreamGoNet(tf.keras.Model, Quantize, XavierOrthogonalInitializer):
    def __init__(
        self,
        *,
        batch_size,
        num_blocks,
        num_dynamics_blocks=None,
        num_channels,
        num_dynamics_channels=None,
        embeddings_size=None,
        policy_coefficient=1.0,
        value_coefficient=1.0,
        ownership_coefficient=0.1,
        similarity_coefficient=0.1,
        num_unrolls=1,
        discount_factor=1.0,
        weight_decay=1e-5,
        label_smoothing=0.2,
        clipnorm=1.0,
        learning_rate_schedule=None,
        lz_weights=None,
        run_eagerly=False
    ):
        super(DreamGoNet, self).__init__()

        self.dg_module = tf.load_op_library('libdg_tf.so')
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.embeddings_size = embeddings_size
        self.policy_coefficient = policy_coefficient
        self.value_coefficient = value_coefficient
        self.ownership_coefficient = ownership_coefficient
        self.similarity_coefficient = similarity_coefficient
        self.num_unrolls = num_unrolls
        self.discount_factor = discount_factor
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate_schedule
        self.lz_weights = lz_weights
        self.features_to_repr = FeaturesToRepr(
            num_blocks=num_blocks,
            num_channels=num_channels,
            embeddings_size=self.embeddings_size
        )
        self.dynamics = Dynamics(
            num_blocks=num_blocks if num_dynamics_blocks is None else num_dynamics_blocks,
            num_channels=num_channels if num_dynamics_channels is None else num_dynamics_channels,
            embeddings_size=self.embeddings_size
        )
        self.rnn = RNN(
            units=self.embeddings_size,
            kernel_initializer=self.xavier_orthogonal_initializer(embeddings_size, embeddings_size),
            recurrent_initializer=self.xavier_orthogonal_initializer(embeddings_size, embeddings_size),
            return_sequences=True,
            time_major=True
        )
        self.predictions = Predictions()

        if self.learning_rate is None:
            self.learning_rate = WarmupExponentialDecaySchedule()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=clipnorm)
        self.swa_optimizer = tfa.optimizers.SWA(self.adam_optimizer)

        # compile the keras model
        self.projection = tf.keras.layers.Dense(units=self.embeddings_size, use_bias=False)
        self.compile(
            run_eagerly=run_eagerly,
            optimizer=tf.keras.mixed_precision.LossScaleOptimizer(
                self.swa_optimizer,
                initial_scale=32768
            )
        )

        # loss metrics
        self.loss_metric = tf.keras.metrics.Mean(name='loss')
        self.loss_policy_metric = tf.keras.metrics.Mean(name='loss/policy')
        self.loss_value_metric = tf.keras.metrics.Mean(name='loss/value')
        self.loss_ownership_metric = tf.keras.metrics.Mean(name='loss/ownership')
        self.loss_similarity_metric = tf.keras.metrics.Mean(name='loss/similarity')
        self.loss_l2_metric = tf.keras.metrics.Mean(name='loss/l2')

        # accuracy metrics
        self.accuracy_policy_1_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='accuracy/policy_1')
        self.accuracy_policy_3_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='accuracy/policy_3')
        self.accuracy_policy_5_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='accuracy/policy_5')
        self.accuracy_value_metric = tf.keras.metrics.Accuracy(name='accuracy/value')
        self.accuracy_ownership_metric = tf.keras.metrics.Accuracy(name='accuracy/ownership')

    @property
    def l2_weights(self):
        return self.features_to_repr.l2_weights + self.dynamics.l2_weights + self.predictions.l2_weights

    def assign_average_vars(self, xs):
        """ Averaging Weights Leads to Wider Optima and Better Generalization [1]

        [1] https://arxiv.org/abs/1803.05407 """

        self.swa_optimizer.assign_average_vars(self.trainable_variables)

        # re-compute batch normalization statistics by walking through the
        # dataset in training mode.
        for (x, labels) in xs:
            self(x, training=True)

    def dump_to(self, out):
        fake_features = tf.ones([1, self.num_unrolls, 19, 19, NUM_FEATURES], tf.float16)
        results = self(fake_features, training=False)

        json.dump(
            {
                'c': {
                    'embeddings_size': self.embeddings_size,
                    'num_features': NUM_FEATURES,
                    'num_repr_channels': self.features_to_repr.num_channels,
                    'num_dyn_channels': self.dynamics.num_channels
                },
                'n': {
                    'r': self.features_to_repr.as_dict(),
                    'd': self.dynamics.as_dict(),
                    'g': self.rnn.as_dict(flat=False),
                    'p': self.predictions.as_dict(),
                },
                't': {
                    'v1': tensor_to_dict(results['value'][0, :]),
                    'v2': tensor_to_dict(results['value'][1, :]),
                    'v3': tensor_to_dict(results['value'][2, :]),
                    'p1': tensor_to_dict(tf.nn.softmax(results['policy'][0, :])),
                    'p2': tensor_to_dict(tf.nn.softmax(results['policy'][1, :])),
                    'p3': tensor_to_dict(tf.nn.softmax(results['policy'][2, :])),
                }
            },
            fp=out,
            sort_keys=True
        )

    def merge_unrolls(self, x):
        shape = tf.shape(x)

        return tf.reshape(
            x,
            tf.concat(
                [
                    [shape[0] * shape[1]],
                    shape[2:]
                ],
                axis=0
            )
        )

    def call(self, inputs, training=True):
        initial_states = self.features_to_repr(inputs[:, 0, :, :, :], training=training)
        embeddings = [
            self.dynamics(inputs[:, i, :, :, :], training=training)
            for i in range(1, self.num_unrolls)
        ]

        # whole_sequence_output is time_major, i.e. [step, batch, embeddings_size]
        whole_sequence_output = self.rnn(
            inputs=self.quantize_and_dequantize(tf.convert_to_tensor(embeddings)),
            initial_state=self.quantize_and_dequantize(initial_states),
            training=training
        )

        whole_sequence_output = tf.concat(
            [
                [initial_states],
                whole_sequence_output
            ],
            axis=0
        )

        # flat_outputs is batch_major, i.e. [batch * step, embeddings_size]
        flat_outputs = tf.transpose(whole_sequence_output, [1, 0, 2])
        flat_outputs = self.merge_unrolls(flat_outputs)

        value_hat, policy_hat, ownership_hat, tower_hat = self.predictions(
            flat_outputs,
            training=training
        )

        return {
            'proj_repr': self.projection(self.features_to_repr(self.merge_unrolls(inputs), training=training)),
            'proj_tower': self.projection(flat_outputs),
            'value': value_hat,
            'policy': policy_hat,
            'ownership': ownership_hat,
            'tower': flat_outputs
        }

    def ownership_loss(self, y_true, y_pred, *, mask):
        y_true = tf.stack([(1 + y_true) / 2, (1 - y_true) / 2], axis=2)  # y_true is -1 or +1 depending on ownership
        y_pred = tf.stack([y_pred, -y_pred], axis=2)
        loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE,
            from_logits=True,
            label_smoothing=self.label_smoothing
        )(
            y_true,
            y_pred
        )
        loss = tf.reduce_mean(input_tensor=loss, axis=[1], keepdims=True)

        # normalize the loss scale for each sample to compensate for the fact
        # that some samples are masked
        mask_scale = tf.cast(tf.size(mask), tf.float32) / (tf.reduce_sum(mask) + 1e-6)

        return tf.reshape(loss * mask * mask_scale, [-1])

    def batch_flatten(self, x):
        return tf.reshape(x, [-1, np.prod(x.shape[1:])])

    def custom_loss(self, y_true, y_pred):
        discounts = tf.reshape(
            tf.repeat(
                tf.constant([pow(self.discount_factor, i) for i in range(self.num_unrolls)], tf.float32),
                tf.size(y_true['value']) // self.num_unrolls,
                axis=0
            ),
            [-1]
        )

        loss_policy = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
            self.merge_unrolls(y_true['policy']),
            y_pred['policy'],
            from_logits=True,
            label_smoothing=self.label_smoothing
        ) * discounts)

        loss_value = tf.reduce_mean(tf.keras.losses.huber(
            self.merge_unrolls(y_true['value']),
            y_pred['value']
        ) * discounts)

        loss_ownership = tf.reduce_mean(self.ownership_loss(
            self.merge_unrolls(y_true['ownership']),
            y_pred['ownership'],
            mask=self.merge_unrolls(y_true['has_ownership'])
        ) * discounts)

        loss_similarity = tf.reduce_mean(
            tf.keras.losses.cosine_similarity(
                tf.cast(tf.stop_gradient(self.batch_flatten(y_pred['proj_repr'])), tf.float32),
                tf.cast(self.batch_flatten(y_pred['proj_tower']), tf.float32)
            )
        )

        loss_l2 = tf.math.accumulate_n(
            [
                tf.nn.l2_loss(weight)
                for weight in self.l2_weights
            ]
        )

        total_loss = self.policy_coefficient * loss_policy \
            + self.value_coefficient * loss_value \
            + self.ownership_coefficient * loss_ownership \
            + self.similarity_coefficient * loss_similarity

        return total_loss, {
            'loss': total_loss,
            'loss/l2': loss_l2,
            'loss/policy': loss_policy,
            'loss/value': loss_value,
            'loss/ownership': loss_ownership,
            'loss/similarity': loss_similarity
        }

    def apply_lz_labels(self, labels):
        if self.lz_weights:
            lz_features = self.merge_unrolls(labels['lz_features'])
            lz_value_hat, lz_policy_hat, lz_tower_hat = leela_zero(lz_features, self.lz_weights)

            labels['value'] = tf.reshape(tf.cast(lz_value_hat, tf.float32), [s if s is not None else -1 for s in labels['value'].shape])
            labels['policy'] = tf.reshape(tf.cast(lz_policy_hat, tf.float32), [s if s is not None else -1 for s in labels['policy'].shape])
            labels['has_ownership'] = tf.zeros_like(labels['has_ownership'])

    @property
    def metrics(self):
        return [
            self.loss_metric,
            self.loss_policy_metric,
            self.loss_value_metric,
            self.loss_ownership_metric,
            self.loss_similarity_metric,
            self.loss_l2_metric,

            self.accuracy_policy_1_metric,
            self.accuracy_policy_3_metric,
            self.accuracy_policy_5_metric,
            self.accuracy_value_metric,
            self.accuracy_ownership_metric
        ]

    def custom_metrics(self, y_true, y_pred, *, losses):
        self.loss_metric.update_state(losses['loss'])
        self.loss_policy_metric.update_state(losses['loss/policy'])
        self.loss_value_metric.update_state(losses['loss/value'])
        self.loss_ownership_metric.update_state(losses['loss/ownership'])
        self.loss_similarity_metric.update_state(losses['loss/similarity'])
        self.loss_l2_metric.update_state(losses['loss/l2'])

        self.accuracy_policy_1_metric.update_state(self.merge_unrolls(y_true['policy']), y_pred['policy'])
        self.accuracy_policy_3_metric.update_state(self.merge_unrolls(y_true['policy']), y_pred['policy'])
        self.accuracy_policy_5_metric.update_state(self.merge_unrolls(y_true['policy']), y_pred['policy'])
        self.accuracy_value_metric.update_state(tf.sign(self.merge_unrolls(y_true['value'])), tf.sign(y_pred['value']))
        self.accuracy_ownership_metric.update_state(tf.sign(self.merge_unrolls(y_true['ownership'])), tf.sign(y_pred['ownership']), sample_weight=tf.repeat(self.merge_unrolls(y_true['has_ownership']), 361, axis=1))

    def custom_image_metrics(self, x, y_true, y_pred):
        def to_heat_image(x, heat):
            return self.dg_module.tensor_to_heat_image(
                x[:, :, 5],
                x[:, :, 17],
                heat
            )

        additional_metrics = {}

        for i in range(self.num_unrolls):
            additional_metrics.update({
                f'ownership/predictions_{i}': to_heat_image(x[0, i, :, :, :], tf.reshape(y_pred['ownership'][i, :], [19, 19])),
                f'ownership/labels_{i}': to_heat_image(x[0, i, :, :, :], tf.reshape(y_true['ownership'][0, i, :], [19, 19])),
                f'policy/predictions_{i}': to_heat_image(x[0, i, :, :, :], tf.reshape(tf.nn.softmax(y_pred['policy'][i, :361]), [19, 19])),
                f'policy/labels_{i}': to_heat_image(x[0, i, :, :, :], tf.reshape(y_true['policy'][0, i, :361], [19, 19]))
            })

        for i in range(self.num_unrolls):
            for j in range(NUM_FEATURES):
                additional_metrics.update({
                    f'features/{i:02}_{j:02}': to_heat_image(x[0, i, :, :, :], tf.cast(x[0, i, :, :, j], tf.float32))
                })

        return additional_metrics

    def custom_gradient(self, tape, loss):
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)

        return gradients, trainable_variables

    def train_step(self, data):
        x, labels = data
        self.apply_lz_labels(labels)

        with tf.GradientTape() as tape:
            y_hat = self(x, training=True)
            loss, losses = self.custom_loss(labels, y_hat)
            loss = self.optimizer.get_scaled_loss(loss)

        gradients, trainable_vars = self.custom_gradient(tape, loss)
        gradients = self.optimizer.get_unscaled_gradients(gradients)
        weight_decay_ops = [var.assign_sub(self.weight_decay * var) for var in self.l2_weights]

        with tf.control_dependencies(weight_decay_ops):
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.custom_metrics(labels, y_hat, losses=losses)

        return {
            m.name: m.result()
            for m in self.metrics
        }

    def test_step(self, data):
        x, labels = data
        self.apply_lz_labels(labels)

        y_hat = self(x, training=False)
        loss, losses = self.custom_loss(labels, y_hat)
        self.custom_metrics(labels, y_hat, losses=losses)
        additional_metrics = self.custom_image_metrics(x, labels, y_hat)
        additional_metrics.update({
            metric.name: metric.result()
            for metric in self.metrics
        })

        return additional_metrics


class CustomTensorBoardCallback(tf.keras.callbacks.Callback):
    """ Custom callback for logging to Tensorboard according to our previous
    established format. """

    def __init__(self, model_dir, *, hparams, early_stopping, learning_rate):
        super(CustomTensorBoardCallback, self).__init__()

        self.early_stopping = early_stopping
        self.learning_rate = learning_rate
        self.writer = tf.summary.create_file_writer(f'{model_dir}')
        self.writer_eval = tf.summary.create_file_writer(f'{model_dir}/eval')
        self.global_step_sec = tf.keras.metrics.Mean('global_step/sec')

        # HParams
        hp_metrics = [
            hp.Metric('accuracy/value', display_name='Value Accuracy'),
            hp.Metric('accuracy/policy_1', display_name='Policy Accuracy (Top 1)'),
            hp.Metric('accuracy/policy_3', display_name='Policy Accuracy (Top 3)'),
            hp.Metric('accuracy/policy_5', display_name='Policy Accuracy (Top 5)'),
        ]

        with self.writer.as_default():
            hp.hparams_config(hparams=hparams.keys(), metrics=hp_metrics)
            hp.hparams(hparams)
        with self.writer_eval.as_default():
            hp.hparams_config(hparams=hparams.keys(), metrics=hp_metrics)
            hp.hparams(hparams)

    def step(self):
        return self.model.optimizer.iterations

    def on_train_batch_begin(self, batch, logs):
        self.batch_start_time = time()

    def on_train_batch_end(self, batch, logs):
        elapsed_sec = time() - self.batch_start_time

        with self.writer.as_default():
            tf.summary.scalar('learning_rate', self.learning_rate(self.step()), step=self.step())
            tf.summary.scalar('loss_scale', self.model.optimizer.loss_scale, step=self.step())

        self.global_step_sec.update_state(1.0 / elapsed_sec)

    def on_test_batch_end(self, batch, logs):
        with self.writer_eval.as_default():
            for name, value in logs.items():
                if isinstance(value, (np.ndarray, np.generic)):
                    tf.summary.image(name, value, step=self.step(), max_outputs=100)
        self.writer_eval.flush()

    def on_epoch_begin(self, epoch, logs=None):
        with self.writer.as_default():
            for weight in self.model.trainable_variables:
                tf.summary.scalar(f'norms/{weight.name}', tf.norm(weight), step=self.step())
            for weight in self.model.non_trainable_weights:
                if weight.dtype == tf.float32 or weight.dtype == tf.float16:
                    tf.summary.scalar(f'norms/{weight.name}', tf.norm(weight), step=self.step())
        self.writer.flush()

    def on_epoch_end(self, epoch, logs):
        for name, value in logs.items():
            if isinstance(value, (int, float)):
                if name.startswith('val_'):
                    with self.writer_eval.as_default():
                        tf.summary.scalar(name[4:], value, step=self.step())
                else:
                    with self.writer.as_default():
                        tf.summary.scalar(name, value, step=self.step())

        # custom metrics
        with self.writer.as_default():
            tf.summary.scalar('global_step/sec', self.global_step_sec.result(), step=self.step())

            if self.early_stopping is not None and self.early_stopping.samples() > 2:
                tf.summary.scalar('learning_rate/p_decreasing', self.early_stopping.is_decreasing(), step=self.step())
                tf.summary.scalar('learning_rate/p_decreasing_90', self.early_stopping.is_decreasing(q=90), step=self.step())
                tf.summary.scalar('learning_rate/slope', self.early_stopping.slope(), step=self.step())
                tf.summary.scalar('learning_rate/slope_90', self.early_stopping.slope(q=90), step=self.step())

        self.writer.flush()
        self.writer_eval.flush()

        # reset statistics
        self.global_step_sec = tf.keras.metrics.Mean('global_step/sec')
