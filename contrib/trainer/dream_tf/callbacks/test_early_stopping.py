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
import unittest

from ..test_common import TestUtils
from .early_stopping import EarlyStoppingCallback, lsq_fit, is_decreasing, percentile

class EarlyStoppingCallbackTest(unittest.TestCase, TestUtils):
    def setUp(self):
        self.model = tf.keras.Model()
        self.early_losses = EarlyStoppingCallback(
            num_warmup_steps=0,
            num_samples=10,
            max_slope=0.0,
            monitor='val_loss'
        )

        self.early_losses.set_model(self.model)

    def test_lsq_fit(self):
        y = [1, 1, 1, 1, 0]
        m, c, y_pred = lsq_fit(y)

        self.assertAlmostEqual(m, -0.2)
        self.assertAlmostEqual(c, 1.4)

    def test_is_decreasing(self):
        y = [1, 1, 1, 1, 0]

        self.assertAlmostEqual(is_decreasing(y, threshold=-5e-6), 0.95836388)

    def test_percentile(self):
        y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.assertTrue(
            np.array_equal(
                percentile(y, 90),
                [1, 2, 3, 4, 5, 6, 7, 8, 9]
            )
        )

    def test_losses_decreasing(self):
        self.early_losses.set_losses([10, 9, 8, 7, 6, 5, 5, 5, 5, 5])
        self.early_losses.check_stopping(iterations=0)

        self.assertFalse(self.model.stop_training)

    def test_losses_increasing(self):
        self.early_losses.set_losses([10, 9, 8, 7, 7, 7, 8, 9, 10, 11])
        self.early_losses.check_stopping(iterations=0)

        self.assertTrue(self.model.stop_training)

if __name__ == '__main__':
    unittest.main()
