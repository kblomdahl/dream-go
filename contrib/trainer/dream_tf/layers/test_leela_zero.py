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
import unittest

from ..test_common import TestUtils
from .leela_zero import leela_zero

class LzTest(unittest.TestCase, TestUtils):
    def setUp(self):
        self.batch_size = 2
        self.x = tf.zeros([self.batch_size, 19, 19, 18], tf.float16)

    def test_shape(self):
        v, p, y = leela_zero(self.x, 'fixtures/d645af9.gz')

        self.assertEqual(v.shape, [self.batch_size, 1])
        self.assertEqual(p.shape, [self.batch_size, 362])
        self.assertEqual(y.shape, [self.batch_size, 19, 19, 8])

if __name__ == '__main__':
    unittest.main()
