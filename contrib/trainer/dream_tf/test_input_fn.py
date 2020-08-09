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

from .input_fn import input_fn
from .layers import NUM_FEATURES

class InputFnTest(unittest.TestCase):
    def setUp(self):
        self.dataset = input_fn('', 2, None, True)

    def tearDown(self):
        tf.reset_default_graph()

    def test_shape(self):
        features, labels = self.dataset.output_shapes

        self.assertEqual(features.as_list(), [None, 19, 19, NUM_FEATURES])
        self.assertEqual(labels['boost'].as_list(), [None, 1])
        self.assertEqual(labels['value'].as_list(), [None, 1])
        self.assertEqual(labels['policy'].as_list(), [None, 362])
        self.assertEqual(labels['next_policy'].as_list(), [None, 362])
        self.assertEqual(labels['ownership'].as_list(), [None, 361])
        self.assertEqual(labels['has_ownership'].as_list(), [None, 1])
        self.assertEqual(labels['komi'].as_list(), [None, 1])

    def test_data_type(self):
        features, labels = self.dataset.output_types

        self.assertEqual(features, tf.float16)
        self.assertEqual(labels['boost'], tf.float32)
        self.assertEqual(labels['value'], tf.float32)
        self.assertEqual(labels['policy'], tf.float32)
        self.assertEqual(labels['next_policy'], tf.float32)
        self.assertEqual(labels['ownership'], tf.float32)
        self.assertEqual(labels['has_ownership'], tf.float32)
        self.assertEqual(labels['komi'], tf.float32)

if __name__ == '__main__':
    unittest.main()
