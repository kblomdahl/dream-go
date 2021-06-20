# Copyright (c) 2020 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
from .value_head import ValueHead

class ValueHeadTest(unittest.TestCase, TestUtils):
    def setUp(self):
        self.batch_size = 1
        self.num_channels = 32
        self.x = tf.zeros([self.batch_size, 19, 19, self.num_channels], tf.float16)
        self.value_head = ValueHead(num_samples=2)

    def test_shape(self):
        value_hat, ownership_hat, value_ownership_hat = self.value_head(self.x, training=True)
        self.assertEqual(value_hat.shape, [self.batch_size, 1])
        self.assertEqual(ownership_hat.shape, [self.batch_size, 361])
        self.assertEqual(value_ownership_hat.shape, [self.batch_size, 361, 2])

    def test_data_type(self):
        value_hat, ownership_hat, value_ownership_hat = self.value_head(self.x, training=True)
        self.assertEqual(value_hat.dtype, tf.float32)
        self.assertEqual(ownership_hat.dtype, tf.float32)
        self.assertEqual(value_ownership_hat.dtype, tf.float32)

    def test_fit(self):
        history = self.fit_regression(
            inputs= \
                np.random.random([1, 19, 19, self.num_channels])
                    .repeat(self.batch_size, axis=0),
            outputs=self.value_head,
            labels= \
                np.random.random([1, 1])
                    .repeat(self.batch_size, axis=0),
        )

        self.assertDecreasing(history)

if __name__ == '__main__':
    unittest.main()
