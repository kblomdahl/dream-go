# Copyright (c) 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

import tensorflow as tf
import numpy as np


class PrettyTensorBoard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(PrettyTensorBoard, self).__init__()

        self.model = None
        self.log_dir = log_dir
        self.writer = tf.compat.v2.summary.create_file_writer(
            log_dir,
            flush_millis=30000
        )

        self._is_tracing = False
        self._total_samples_seen = 0
        self._epoch_logs = {}
        self._chief_worker_only = True

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        self.writer.set_as_default()

    def on_epoch_begin(self, epoch, logs=None):
        for batch_log in self._epoch_logs.values():
            batch_log.clear()

    def on_train_batch_begin(self, batch, logs=None):
        # always trace the second batch in each epoch
        if batch == 1 and tf.executing_eagerly():
            tf.compat.v2.summary.trace_on(graph=True, profiler=True)
            self._is_tracing = True

        tf.compat.v2.summary.experimental.set_step(self._total_samples_seen)

    def on_train_batch_end(self, batch, logs=None):
        for metric, value in logs.items():
            if metric in ['batch', 'size']:
                continue

            metric = _get_metric_name(metric)
            self._epoch_logs.setdefault(metric, []).append(value)
            if self._total_samples_seen % 50 == 0:
                tf.summary.scalar(metric, value)

        self._total_samples_seen += logs.get('size', 1)

        # if there is a trace running, then export its output
        if self._is_tracing:
            self._is_tracing = False

            tf.compat.v2.summary.trace_export(
                'trace',
                step=self._total_samples_seen,
                profiler_outdir=self.log_dir
            )

    def on_epoch_end(self, epoch, logs):
        for metric in self.model.metrics:
            metric.reset_states()

        for metric, samples in self._epoch_logs.items():
            try:
                slope, offset = _get_slope(samples)

                tf.summary.scalar('loss_slope/' + metric, slope)
                tf.summary.scalar('loss_offset/' + metric, offset)
            except ValueError as e:
                print(e)

        #for layer in self.model.layers:
        #    for weight in layer.weights:
        #        tf.summary.histogram('weights' + weight.name, weight)


ACCURACY_SUFFICES = (
    '_binary_accuracy',
    '_categorical_accuracy',
    '_accuracy'
)

LOSS_SUFFICES = (
    '_loss',
)


def _get_metric_name(name, prefix=None):
    for suffix in ACCURACY_SUFFICES:
        if name.endswith(suffix):
            return (prefix or 'accuracy/') + name[:-len(suffix)]

    for suffix in LOSS_SUFFICES:
        if name.endswith(suffix):
            return (prefix or 'loss/') + name[:-len(suffix)]

    if name == 'loss':
        return (prefix or 'loss/') + 'loss'

    return name


def _get_slope(samples):
    """ Estimate the slope of the given samples by fitting a linear function to
    the provided samples using least squares """
    if len(samples) < 2:
        raise ValueError('Insufficient sample count to determine the slope')

    num_samples = len(samples)
    a = np.ones((num_samples, 2))
    a[:, 0] = np.linspace(-1.0, 0.0, num=num_samples)

    try:
        [slope, offset], _residuals, rank, _s = np.linalg.lstsq(a, samples, rcond=-1)
        assert rank == 2

        return slope, offset
    except np.linalg.LinAlgError as e:
        raise ValueError(e)
