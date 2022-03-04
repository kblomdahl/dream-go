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
from .mixer_block import MlpBlock, MixerBlock

class MlpBlockTest(TestUtils, unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.x = tf.zeros([self.batch_size, 361, 128], tf.float16)
        self.mlp_block = MlpBlock(256)

    def test_data_type(self):
        y = self.mlp_block(self.x)
        self.assertEqual(y.shape, self.x.shape)

    def test_shape(self):
        y = self.mlp_block(self.x)
        self.assertEqual(y.dtype, self.x.dtype)

class MixerBlockTest(TestUtils, unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_channels = 128
        self.x = tf.zeros([self.batch_size, 19, 19, self.num_channels], tf.float16)
        self.mixer_block = MixerBlock(tokens_mlp_dims=361, channels_mlp_dims=self.num_channels)

    def test_data_type(self):
        y = self.mixer_block(self.x)
        self.assertEqual(y.shape, self.x.shape)

    def test_shape(self):
        y = self.mixer_block(self.x)
        self.assertEqual(y.dtype, self.x.dtype)

    def test_fit(self):
        history = self.fit_regression(
            inputs= \
                np.random.random([1, 19, 19, self.num_channels])
                    .repeat(self.batch_size, axis=0),
            outputs=self.mixer_block,
            labels= \
                np.random.random([1, 19, 19, self.num_channels])
                    .repeat(self.batch_size, axis=0)
        )

        self.assertDecreasing(history)

if __name__ == '__main__':
    unittest.main()
