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

import tensorflow as tf
import numpy as np
import scipy.stats

from collections import deque

class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        *,
        monitor='val_loss',
        num_warmup_steps=5000,
        num_samples=50,
        max_slope=-1e-7
    ):
        super(EarlyStoppingCallback, self).__init__()

        self.monitor = monitor
        self.num_warmup_steps = num_warmup_steps
        self.max_slope = max_slope
        self.losses = deque(maxlen=num_samples)

    def set_losses(self, losses):
        self.losses = losses  # for testing purposes

    def on_epoch_end(self, epoch, logs):
        self.losses.append(logs[self.monitor])
        self.check_stopping()

    def check_stopping(self, iterations=None):
        if iterations is None:
            iterations = self.model.optimizer.iterations.numpy()
        if iterations >= self.num_warmup_steps:
            if self.is_decreasing(q=100) < 0.51 and self.is_decreasing(q=90) < 0.51:
                self.model.stop_training = True

    def samples(self, q=100):
        return len(percentile(self.losses, q))

    def slope(self, q=100):
        return lsq_fit(percentile(self.losses, q))[0]

    def is_decreasing(self, q=100):
        return is_decreasing(percentile(self.losses, q), threshold=self.max_slope)


def percentile(losses, q):
    """ Returns all elements from the given array that are less than the q:th
    percentile. """

    if q == 100:
        return losses
    else:
        losses = np.array(losses)
        threshold = np.percentile(losses, q)
        return losses[losses < threshold]


def lsq_fit(losses):
    """ Returns the coefficients of a linear polynomial fit to the given
    points. """

    n = len(losses)
    y = np.array(losses)
    x = np.stack([np.arange(n) + 1, np.ones_like(y)], axis=-1)
    m, c = np.linalg.lstsq(x, y, rcond=None)[0]

    return m, c, np.matmul(x, [m, c])


def is_decreasing(losses, threshold=-5e-6):
    """ Returns the probability that the given points are decreasing by at least
    the given threshold for each step. """

    n = len(losses)
    if n <= 2:
        return 1.0
    m, c, y_pred = lsq_fit(losses)

    square_error = np.sum(np.square(y_pred - losses))
    if square_error == 0.0:
        variance = 0.0
    else:
        variance = 1.0 / (n - 2.0) * square_error
        variance = (12.0 * variance) / (n ** 3 - n)

    return scipy.stats.norm.cdf(threshold, loc=m, scale=np.sqrt(variance))
