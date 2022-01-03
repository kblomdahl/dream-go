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
from .dynamics import Dynamics
from . import NUM_FEATURES

class DynamicsTest(unittest.TestCase, TestUtils):
    def setUp(self):
        self.batch_size = 2
        self.num_channels = 32
        self.num_output_channels = 16
        self.x = tf.zeros([self.batch_size, 19, 19, self.num_output_channels], tf.float16)
        self.state = tf.zeros([self.batch_size, 19, 19, NUM_FEATURES], tf.float16)
        self.layer = Dynamics(num_blocks=6, num_channels=self.num_channels)

    def test_shape(self):
        y = self.layer([self.x, self.state])

        self.assertEqual(y.shape, [self.batch_size, 19, 19, self.num_output_channels])

    def test_dtype(self):
        y = self.layer([self.x, self.state])

        self.assertEqual(y.dtype, tf.float16)

if __name__ == '__main__':
    unittest.main()
