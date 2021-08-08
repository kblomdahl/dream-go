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

from ..test_common import TestUtils
from .predictions import Predictions

class PredictionsTest(unittest.TestCase, TestUtils):
    def setUp(self):
        self.batch_size = 2
        self.num_channels = 16
        self.x = tf.zeros([self.batch_size, 19, 19, self.num_channels], tf.float16)
        self.layer = Predictions()

    def test_shape(self):
        v, vy, p, vo, x = self.layer(self.x)

        self.assertEqual(v.shape, [self.batch_size, 1])
        self.assertEqual(vy.shape, [self.batch_size, 361, 2])
        self.assertEqual(p.shape, [self.batch_size, 362])
        self.assertEqual(vo.shape, [self.batch_size, 361])
        self.assertEqual(x.shape, [self.batch_size, 19, 19, self.num_channels])

    def test_dtype(self):
        v, vy, p, vo, x = self.layer(self.x)

        self.assertEqual(v.dtype, tf.float32)
        self.assertEqual(vy.dtype, tf.float32)
        self.assertEqual(p.dtype, tf.float32)
        self.assertEqual(vo.dtype, tf.float32)
        self.assertEqual(x.dtype, tf.float16)

if __name__ == '__main__':
    unittest.main()
