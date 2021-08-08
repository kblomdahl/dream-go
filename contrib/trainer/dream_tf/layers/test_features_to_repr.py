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

import unittest

import tensorflow as tf
import numpy as np

from ..test_common import TestUtils
from .features_to_repr import FeaturesToRepr

class FeaturesToReprTest(unittest.TestCase, TestUtils):
    def setUp(self):
        self.batch_size = 2
        self.num_channels = 16
        self.num_output_channels = 8
        self.x = tf.zeros([self.batch_size, 19, 19, self.num_channels], tf.float16)
        self.layer = FeaturesToRepr(num_blocks=2, num_channels=self.num_channels, num_output_channels=self.num_output_channels)

    def test_shape(self):
        y = self.layer(self.x)

        self.assertEqual(y.shape, [self.batch_size, 19, 19, self.num_output_channels])

    def test_dtype(self):
        y = self.layer(self.x)

        self.assertEqual(y.dtype, tf.float16)

    def test_fit(self):
        history = self.fit_regression(
            inputs= \
                np.random.random([1, 19, 19, self.num_channels])
                    .repeat(self.batch_size, axis=0),
            outputs=self.layer,
            labels= \
                np.random.random([1, 19, 19, self.num_output_channels])
                    .repeat(self.batch_size, axis=0)
        )

        self.assertDecreasing(history)

if __name__ == '__main__':
    unittest.main()
