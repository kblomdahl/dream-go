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
import unittest

from . import NUM_FEATURES
from ..test_common import TestUtils
from .tower import Tower

class TowerTest(unittest.TestCase, TestUtils):
    def setUp(self):
        self.batch_size = 16
        self.x = tf.zeros([self.batch_size, 19, 19, NUM_FEATURES], tf.float16)

    def test_shape(self):
        v, vo, p, o, y = Tower(num_blocks=6, num_channels=64)(self.x, training=True)
        self.assertEqual(v.shape, [self.batch_size, 1])
        self.assertEqual(vo.shape, [self.batch_size, 361, 2])
        self.assertEqual(p.shape, [self.batch_size, 362])
        self.assertEqual(o.shape, [self.batch_size, 361])
        self.assertEqual(y.shape, [self.batch_size, 19, 19, 64])


if __name__ == '__main__':
    unittest.main()
