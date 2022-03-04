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

import unittest

import tensorflow as tf

from ..test_common import TestUtils
from .batch_norm import BatchNormConv2D, BatchNormDense

class BatchNormConv2DTestBase(TestUtils):
    def setUp(self):
        self.x = tf.zeros([16, 19, 19, 64], tf.float16)
        self.conv = BatchNormConv2D()

    def test_dtype(self):
        self.assertEqual(
            self.conv(self.x, training=self.training).dtype,
            self.x.dtype
        )

    def test_shape(self):
        self.assertEqual(
            self.conv(self.x, training=self.training).shape,
            self.x.shape
        )

class BatchNormConv2DEvalTest(BatchNormConv2DTestBase, unittest.TestCase):
    @property
    def training(self):
        return False

class BatchNormConv2DTrainTest(BatchNormConv2DTestBase, unittest.TestCase):
    @property
    def training(self):
        return True

class BatchNormDenseTestBase(TestUtils):
    def setUp(self):
        self.batch_size = 4
        self.out_dims = 16
        self.x = tf.zeros([self.batch_size, 361, 64], tf.float16)
        self.dense = BatchNormDense(out_dims=self.out_dims)

    def test_dtype(self):
        self.assertEqual(
            self.dense(self.x, training=self.training).dtype,
            self.x.dtype
        )

    def test_shape(self):
        self.assertEqual(
            self.dense(self.x, training=self.training).shape,
            [self.batch_size, 361, self.out_dims]
        )

class BatchNormDenseEvalTest(BatchNormDenseTestBase, unittest.TestCase):
    @property
    def training(self):
        return False

class BatchNormDenseTrainTest(BatchNormDenseTestBase, unittest.TestCase):
    @property
    def training(self):
        return True

if __name__ == '__main__':
    unittest.main()
