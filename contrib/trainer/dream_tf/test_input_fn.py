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

class CommonInputFnTest:
    def tearDown(self):
        tf.compat.v1.reset_default_graph()

    def test_shape(self):
        features, labels = self.dataset.element_spec

        self.assertEqual(features.shape.as_list(), [None, 19, 19, NUM_FEATURES])
        self.assertEqual(labels['lz_features'].shape.as_list(), [None, 19, 19, 18])
        self.assertEqual(labels['boost'].shape.as_list(), [None, 1])
        self.assertEqual(labels['value'].shape.as_list(), [None, 1])
        self.assertEqual(labels['policy'].shape.as_list(), [None, 362])
        self.assertEqual(labels['next_policy'].shape.as_list(), [None, 362])
        self.assertEqual(labels['ownership'].shape.as_list(), [None, 361])
        self.assertEqual(labels['has_ownership'].shape.as_list(), [None, 1])
        self.assertEqual(labels['komi'].shape.as_list(), [None, 1])

    def test_data_type(self):
        features, labels = self.dataset.element_spec

        self.assertEqual(features.dtype, tf.float16)
        self.assertEqual(labels['lz_features'].dtype, tf.float16)
        self.assertEqual(labels['boost'].dtype, tf.float32)
        self.assertEqual(labels['value'].dtype, tf.float32)
        self.assertEqual(labels['policy'].dtype, tf.float32)
        self.assertEqual(labels['next_policy'].dtype, tf.float32)
        self.assertEqual(labels['ownership'].dtype, tf.float32)
        self.assertEqual(labels['has_ownership'].dtype, tf.float32)
        self.assertEqual(labels['komi'].dtype, tf.float32)

class InputFnTrainingModeTest(unittest.TestCase, CommonInputFnTest):
    def setUp(self):
        self.dataset = input_fn('', 2, None, True)

class InputFnTestModeTest(unittest.TestCase, CommonInputFnTest):
    def setUp(self):
        self.dataset = input_fn('', 2, None, False)


if __name__ == '__main__':
    unittest.main()
