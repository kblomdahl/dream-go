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

import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp

from .layers.leela_zero import leela_zero
from .layers.tower import Tower
from .optimizers.schedules.learning_rate_schedule import WarmupExponentialDecaySchedule

class DreamGoNet(tf.keras.Model):
    def __init__(
        self,
        *,
        num_blocks,
        num_channels,
        num_policy_channels=8,
        num_value_channels=2,
        weight_decay=1e-5,
        label_smoothing=0.2,
        learning_rate_schedule=None,
        lz_weights=None
    ):
        super(DreamGoNet, self).__init__()

        self.num_channels = num_channels
        self.num_policy_channels = num_policy_channels
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate_schedule
        self.lz_weights = lz_weights
        self.tower = Tower(
            num_blocks=num_blocks,
            num_channels=num_channels,
            num_policy_channels=num_policy_channels,
            num_value_channels=num_value_channels
        )

        if self.learning_rate is None:
            self.learning_rate = WarmupExponentialDecaySchedule()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizer = tfa.optimizers.SWA(self.optimizer)

        # compile the keras model
        self.compile(optimizer=self.optimizer)

        # loss metrics
        self.loss_metric = tf.keras.metrics.Mean(name='loss')
        self.loss_policy_metric = tf.keras.metrics.Mean(name='loss/policy')
        self.loss_value_metric = tf.keras.metrics.Mean(name='loss/value')
        self.loss_ownership_metric = tf.keras.metrics.Mean(name='loss/ownership')
        self.loss_l2_metric = tf.keras.metrics.Mean(name='loss/l2')

        # accuracy metrics
        self.accuracy_policy_1_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='accuracy/policy_1')
        self.accuracy_policy_3_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='accuracy/policy_3')
        self.accuracy_policy_5_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='accuracy/policy_5')
        self.accuracy_value_metric = tf.keras.metrics.Accuracy(name='accuracy/value')
        self.accuracy_ownership_metric = tf.keras.metrics.Accuracy(name='accuracy/ownership')

    def assign_average_vars(self, xs):
        """ Averaging Weights Leads to Wider Optima and Better Generalization [1]

        [1] https://arxiv.org/abs/1803.05407 """

        self.optimizer.assign_average_vars(self.tower.get_weights())

        # re-compute batch normalization statistics by walking through the
        # dataset in training mode.
        for (x, _) in xs:
            self.tower(x, training=True)

    def dump_to(self, out):
        json.dump(
            {
                'num_channels:0': self.num_channels,
                'num_samples:0': self.num_policy_channels,
                **self.tower.as_dict()
            },
            fp=out,
            sort_keys=True
        )

    def call(self, inputs, training=True):
        value_hat, value_ownership_hat, policy_hat, ownership_hat, tower_hat = self.tower(
            inputs,
            training=training
        )

        return {
            'value': value_hat,
            'value_ownership': value_ownership_hat,
            'policy': policy_hat,
            'ownership': ownership_hat,
            'tower': tower_hat
        }

    def ownership_loss(self, y_true, y_pred, *, mask):
        y_true = tf.stack([(1 + y_true) / 2, (1 - y_true) / 2], axis=2)
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

    def custom_loss(self, y_true, y_pred):
        loss_policy = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
            y_true['policy'],
            y_pred['policy'],
            from_logits=True,
            label_smoothing=self.label_smoothing
        ))

        loss_value = tf.reduce_mean(tf.keras.losses.huber(
            y_true['value'],
            y_pred['value']
        ))

        loss_ownership = tf.reduce_mean(self.ownership_loss(
            y_true['ownership'],
            y_pred['ownership'],
            mask=y_true['has_ownership']
        ))

        loss_l2 = tf.math.accumulate_n(
            [
                tf.nn.l2_loss(weight)
                for weight in self.tower.l2_weights
            ]
        )

        total_loss = 0.12 * loss_policy \
            + 1.00 * loss_value \
            + 1.00 * loss_ownership

        return total_loss, {
            'loss': total_loss,
            'loss/l2': loss_l2,
            'loss/policy': loss_policy,
            'loss/value': loss_value,
            'loss/ownership': loss_ownership
        }

    def apply_lz_labels(self, labels):
        if self.lz_weights:
            lz_value_hat, lz_policy_hat, lz_tower_hat = leela_zero(labels['lz_features'], self.lz_weights)

            labels['value'] = tf.cast(lz_value_hat, tf.float32)
            labels['policy'] = tf.cast(lz_policy_hat, tf.float32)
            labels['has_ownership'] = tf.zeros_like(labels['has_ownership'])

    def get_scaled_loss(self, loss):
        return 128.0 * loss

    def get_unscaled_gradients(self, gradients):
        recip_loss_scale = 1 / 128.0

        return list([
            recip_loss_scale * gradient
            for gradient in gradients
        ])

    @property
    def metrics(self):
        return [
            self.loss_metric,
            self.loss_policy_metric,
            self.loss_value_metric,
            self.loss_ownership_metric,
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
        self.loss_l2_metric.update_state(losses['loss/l2'])

        self.accuracy_policy_1_metric.update_state(y_true['policy'], y_pred['policy'])
        self.accuracy_policy_3_metric.update_state(y_true['policy'], y_pred['policy'])
        self.accuracy_policy_5_metric.update_state(y_true['policy'], y_pred['policy'])
        self.accuracy_value_metric.update_state(tf.sign(y_true['value']), tf.sign(y_pred['value']))
        self.accuracy_ownership_metric.update_state(tf.sign(y_true['ownership']), tf.sign(y_pred['ownership']), sample_weight=tf.repeat(y_true['has_ownership'], 361, axis=1))

    def train_step(self, data):
        x, labels = data
        self.apply_lz_labels(labels)

        with tf.GradientTape() as tape:
            y_hat = self(x, training=True)
            loss, losses = self.custom_loss(labels, y_hat)
            loss = self.get_scaled_loss(loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        gradients = self.get_unscaled_gradients(gradients)
        weight_decay_ops = [var.assign_sub(self.weight_decay * var) for var in self.tower.l2_weights]

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

        return {
            metric.name: metric.result()
            for metric in self.metrics
        }


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

        self.global_step_sec.update_state(1.0 / elapsed_sec)

    def on_epoch_end(self, epoch, logs):
        for name, value in logs.items():
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

        # reset statistics
        self.global_step_sec = tf.keras.metrics.Mean('global_step/sec')
