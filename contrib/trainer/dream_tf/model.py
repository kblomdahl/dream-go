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
from tensorboard.plugins.hparams import api as hp

from .layers.leela_zero import LeelaZero
from .models.dynamics_model import DynamicsModel
from .models.predictor import Predictor
from .models.representation_model import RepresentationModel
from .models.target_predictor import TargetPredictorModel

class DreamGoNet(tf.keras.Model):
    def __init__(
        self,
        *,
        batch_size,
        num_stoch_channels,
        num_repr_blocks,
        num_repr_channels,
        num_dyn_blocks,
        num_dyn_channels,
        num_pred_layers,
        policy_coefficient=1.0,
        value_coefficient=1.0,
        target_coefficient=0.1,
        similarity_coefficient=0.1,
        num_unrolls=1,
        discount_factor=1.0,
        weight_decay=1e-5,
        label_smoothing=0.2,
        clipnorm=1.0,
        learning_rate=1e-4,
        lz_weights=None,
        run_eagerly=False
    ):
        super(DreamGoNet, self).__init__()

        self.dg_module = tf.load_op_library('libdg_tf.so')
        self.batch_size = batch_size
        self.num_unrolls = num_unrolls
        self.policy_coefficient = policy_coefficient
        self.value_coefficient = value_coefficient
        self.target_coefficient = target_coefficient
        self.similarity_coefficient = similarity_coefficient
        self.discount_factor = discount_factor
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.leela_zero = LeelaZero(lz_weights) if lz_weights else None
        self.num_features = int(self.dg_module.num_feature_channels())
        self.num_motion_features = int(self.dg_module.num_motion_channels())
        self.num_targets = int(self.dg_module.num_target_channels())
        self.num_stoch_channels = num_stoch_channels
        self.representation_model = RepresentationModel(
            num_blocks=num_repr_blocks,
            num_channels=num_repr_channels,
            num_out_channels=num_stoch_channels
        )
        self.dynamics_model = DynamicsModel(
            num_blocks=num_dyn_blocks,
            num_channels=num_dyn_channels,
            num_out_channels=num_stoch_channels
        )
        self.predictor = Predictor(
            layers=num_pred_layers,
            embeddings_size=361 * num_stoch_channels,
            output_shape=[self.batch_size, self.num_unrolls, -1]
        )
        self.target_predictor = TargetPredictorModel(
            num_blocks=num_repr_blocks,
            num_channels=num_repr_channels,
            num_targets=self.num_targets,
            output_shape=[self.batch_size, self.num_unrolls, 19, 19, self.num_targets]
        )

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)

        # compile the keras model
        self.compile(
            run_eagerly=run_eagerly,
            optimizer=tf.keras.mixed_precision.LossScaleOptimizer(
                self.adam_optimizer,
                initial_scale=2048
            )
        )

        # loss metrics
        self.loss_metric = tf.keras.metrics.Mean(name='loss')
        self.loss_policy_metric = tf.keras.metrics.Mean(name='loss/policy')
        self.loss_value_metric = tf.keras.metrics.Mean(name='loss/value')
        self.loss_similarity_metric = tf.keras.metrics.Mean(name='loss/similarity')
        self.loss_targets_metric = tf.keras.metrics.Mean(name='loss/targets')
        self.loss_l2_metric = tf.keras.metrics.Mean(name='loss/l2')

        # accuracy metrics
        self.accuracy_policy_metrics = [tf.keras.metrics.TopKCategoricalAccuracy(k=1, name=f'policy/accuracy/[{i}]') for i in range(self.num_unrolls)]
        self.accuracy_value_metrics = [tf.keras.metrics.Accuracy(name=f'value/accuracy/[{i}]') for i in range(self.num_unrolls)]

    def dump_to(self, out):
        json.dump(
            {
                'c': {
                    'embeddings_size': 361 * self.num_stoch_channels
                },
                'n': {
                    # pass
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

    def call(self, inputs, training=True, labels=None):
        if labels is not None:
            z_true = tf.reshape(self.representation_model(
                self.merge_unrolls(labels['features']),
                training=training
            ), labels['features'].shape[:2] + [19, 19, self.num_stoch_channels])
        else:
            z_true = None
        z_pred = [
            self.representation_model(inputs[:, 0, :, :, :self.num_features], training=training)
        ]

        for i in range(1, self.num_unrolls):
            z_pred.append(
                self.dynamics_model([inputs[:, i, :, :, :self.num_motion_features], z_pred[-1]], training=training)
            )

        # all of the output tensors are in format `[step, batch, ...]`, but the
        # labels are in `[batch, step, ...]`
        z_pred = tf.stack(z_pred, axis=1)

        # use the representation during training, inspired by DreamerV2, instead
        # of the latent convolution.
        z = self.merge_unrolls(z_true if z_true is not None else z_pred)
        value, policy = self.predictor(z, training=training)
        targets = self.target_predictor(z, training=training)

        return {
            'value': value,
            'policy': policy,
            'targets': targets,
            'z_true': z_true,
            'z_pred': z_pred,
        }

    def custom_loss(self, y_true, y_pred):
        discounts = tf.constant(
            [self.discount_factor ** i for i in range(self.num_unrolls)]
        )

        loss_policy = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
            tf.cast(y_true['policy'], tf.float32),
            tf.cast(y_pred['policy'], tf.float32),
            from_logits=True,
            label_smoothing=self.label_smoothing
        ) * discounts)

        loss_value = tf.reduce_mean(tf.keras.losses.huber(
            tf.cast(y_true['value'], tf.float32),
            tf.cast(y_pred['value'], tf.float32)
        ) * discounts)

        if y_pred['z_true'] is not None:
            z_true = tf.reshape(y_pred['z_true'], [self.batch_size, self.num_unrolls, -1])
            z_pred = tf.reshape(y_pred['z_pred'], [self.batch_size, self.num_unrolls, -1])
            loss_similarity = tf.reduce_mean((
                0.8 * tf.keras.losses.cosine_similarity(
                    tf.cast(tf.stop_gradient(z_true), tf.float32),
                    tf.cast(z_pred, tf.float32),
                ) +
                0.2 * tf.keras.losses.cosine_similarity(
                    tf.cast(tf.stop_gradient(z_pred), tf.float32),
                    tf.cast(z_true, tf.float32),
                )
            ) * discounts)
        else:
            loss_similarity = tf.zeros_like(loss_value)

        loss_targets = tf.reduce_mean(tf.keras.losses.cosine_similarity(
            tf.cast(tf.transpose(
                tf.reshape(y_true['targets'], [self.batch_size, self.num_unrolls, -1, self.num_targets]),
                [0, 1, 3, 2]
            ), tf.float32),
            tf.cast(tf.transpose(
                tf.reshape(y_pred['targets'], [self.batch_size, self.num_unrolls, -1, self.num_targets]),
                [0, 1, 3, 2]
            ), tf.float32),
        ) * tf.reshape(discounts, [1, -1, 1]) * tf.cast(y_true['targets_mask'], tf.float32))

        total_loss = self.policy_coefficient * loss_policy \
                + self.value_coefficient * loss_value \
                + self.similarity_coefficient * loss_similarity \
                + self.target_coefficient * loss_targets

        return total_loss, {
            'loss': total_loss,
            'loss/policy': loss_policy,
            'loss/value': loss_value,
            'loss/similarity': loss_similarity,
            'loss/targets': loss_targets,
            'loss/l2': tf.math.accumulate_n([tf.nn.l2_loss(v) for v in self.trainable_variables])
        }

    def apply_lz_labels(self, labels):
        if self.leela_zero:
            lz_features = self.merge_unrolls(labels['lz_features'])
            lz_value_hat, lz_policy_hat = self.leela_zero(lz_features, training=False)

            labels['value'] = tf.reshape(tf.cast(lz_value_hat, tf.float32), [s if s is not None else -1 for s in labels['value'].shape])
            labels['policy'] = tf.reshape(tf.cast(lz_policy_hat, tf.float32), [s if s is not None else -1 for s in labels['policy'].shape])

    @property
    def metrics(self):
        return [
            self.loss_metric,
            self.loss_policy_metric,
            self.loss_value_metric,
            self.loss_similarity_metric,
            self.loss_targets_metric,
            self.loss_l2_metric,

            *self.accuracy_policy_metrics,
            *self.accuracy_value_metrics
        ]

    def custom_metrics(self, y_true, y_pred, *, losses):
        self.loss_metric.update_state(losses['loss'])
        self.loss_policy_metric.update_state(losses['loss/policy'])
        self.loss_value_metric.update_state(losses['loss/value'])
        self.loss_similarity_metric.update_state(losses['loss/similarity'])
        self.loss_targets_metric.update_state(losses['loss/targets'])
        self.loss_l2_metric.update_state(losses['loss/l2'])

        for i in range(self.num_unrolls):
            self.accuracy_policy_metrics[i].update_state(
                y_true['policy'][:, i, :],
                y_pred['policy'][:, i, :]
            )
            self.accuracy_value_metrics[i].update_state(
                tf.sign(y_true['value'][:, i, :]),
                tf.sign(y_pred['value'][:, i, :])
            )

    def custom_image_metrics(self, x, y_true, y_pred):
        def to_heat_image(x, heat):
            return self.dg_module.tensor_to_heat_image(
                x[:, :, 0],
                x[:, :, 8],
                heat
            )

        additional_metrics = {}
        lz = y_true['lz_features']

        for i in range(self.num_unrolls):
            additional_metrics.update({
                f'policy/[{i}]/true': to_heat_image(lz[0, i, :, :, :], tf.reshape(y_true['policy'][0, i, :361], [19, 19])),
                f'policy/[{i}]/pred': to_heat_image(lz[0, i, :, :, :], tf.reshape(tf.nn.softmax(y_pred['policy'][0, i, :361]), [19, 19]))
            })

        for i in range(self.num_unrolls):
            num_features_channels = y_true['features'].shape.as_list()[-1]
            num_motion_channels = y_true['motion_features'].shape.as_list()[-1]

            for j in range(num_features_channels if i == 0 else num_motion_channels):
                additional_metrics.update({
                    f'inputs/[{i:02}]/[{j:02}]': to_heat_image(lz[0, i, :, :, :], tf.cast(x[0, i, :, :, j], tf.float32))
                })

            for j in range(num_features_channels):
                additional_metrics.update({
                    f'features/[{i:02}]/[{j:02}]': to_heat_image(lz[0, i, :, :, :], tf.cast(y_true['features'][0, i, :, :, j], tf.float32))
                })

            for j in range(num_motion_channels):
                additional_metrics.update({
                    f'motion_features/[{i:02}]/[{j:02}]': to_heat_image(lz[0, i, :, :, :], tf.cast(y_true['motion_features'][0, i, :, :, j], tf.float32))
                })

            for j in range(18):
                additional_metrics.update({
                    f'lz_features/[{i:02}]/[{j:02}]': to_heat_image(lz[0, i, :, :, :], tf.cast(y_true['lz_features'][0, i, :, :, j], tf.float32))
                })

            for j in range(y_true['targets'].shape.as_list()[-1]):
                additional_metrics.update({
                    f'targets/[{i:02}]/[{j:02}]/true': to_heat_image(lz[0, i, :, :, :], tf.cast(y_true['targets'][0, i, :, :, j], tf.float32)),
                    f'targets/[{i:02}]/[{j:02}]/pred': to_heat_image(lz[0, i, :, :, :], tf.cast(y_pred['targets'][0, i, :, :, j], tf.float32))
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
            y_hat = self(x, labels=labels, training=True)
            loss, losses = self.custom_loss(labels, y_hat)
            loss = self.optimizer.get_scaled_loss(loss)

        gradients, trainable_vars = self.custom_gradient(tape, loss)
        gradients = self.optimizer.get_unscaled_gradients(gradients)
        weight_decay_ops = [var.assign_sub(self.weight_decay * var) for var in self.trainable_variables]

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

    def __init__(self, model_dir, *, hparams, early_stopping):
        super(CustomTensorBoardCallback, self).__init__()

        self.early_stopping = early_stopping
        self.writer = tf.summary.create_file_writer(f'{model_dir}')
        self.writer_eval = tf.summary.create_file_writer(f'{model_dir}/eval')

        # HParams
        hparams = { key: self._map_hparam(value) for key, value in hparams.items() }
        hp_metrics = [
            hp.Metric('value/accuracy/[0]', display_name='Value Accuracy (%)'),
            hp.Metric('policy/accuracy/[0]', display_name='Policy Accuracy (%)'),
        ]

        with self.writer.as_default():
            hp.hparams_config(hparams=hparams.keys(), metrics=hp_metrics)
            hp.hparams(hparams)
        with self.writer_eval.as_default():
            hp.hparams_config(hparams=hparams.keys(), metrics=hp_metrics)
            hp.hparams(hparams)

    def _map_hparam(self, value):
        if type(value) is list:
            return ', '.join(value)
        elif value is None:
            return ''
        else:
            return value

    def step(self):
        return self.model.optimizer.iterations * self.model.batch_size * self.model.num_unrolls

    def on_train_batch_begin(self, batch, logs):
        self.batch_start_time = time()

    def on_train_batch_end(self, batch, logs):
        elapsed_sec = time() - self.batch_start_time

        with self.writer.as_default():
            tf.summary.scalar('learning_rate', self.model.optimizer.lr, step=self.step())
            tf.summary.scalar('learning_rate/scale', self.model.optimizer.loss_scale, step=self.step())
            tf.summary.scalar('global_step/sec', elapsed_sec, step=self.step())

    def on_test_batch_end(self, batch, logs):
        with self.writer_eval.as_default():
            for name, value in logs.items():
                if isinstance(value, (np.ndarray, np.generic)):
                    tf.summary.image(name, value, step=self.step(), max_outputs=100)
        self.writer_eval.flush()

    def on_epoch_begin(self, epoch, logs=None):
        with self.writer.as_default():
            tf.summary.scalar('learning_rate/epoch', epoch, step=self.step())

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
            if self.early_stopping is not None and self.early_stopping.samples() > 2:
                tf.summary.scalar('learning_rate/p_decreasing', self.early_stopping.is_decreasing(), step=self.step())
                tf.summary.scalar('learning_rate/p_decreasing_90', self.early_stopping.is_decreasing(q=90), step=self.step())
                tf.summary.scalar('learning_rate/slope', self.early_stopping.slope(), step=self.step())
                tf.summary.scalar('learning_rate/slope_90', self.early_stopping.slope(q=90), step=self.step())

        self.writer.flush()
        self.writer_eval.flush()
