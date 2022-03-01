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

from .input_fn import input_fn
from .test_common import TestUtils

class CommonInputFnTest(TestUtils):
    def test_shape(self):
        for features, labels in self.dataset:
            self.assertEqual(features.shape.as_list(), [self.batch_size, self.num_unrolls, 19, 19, self.num_feature_channels])
            self.assertEqual(labels['features'].shape.as_list(), [self.batch_size, self.num_unrolls, 19, 19, self.num_feature_channels])
            self.assertEqual(labels['motion_features'].shape.as_list(), [self.batch_size, self.num_unrolls, 19, 19, self.num_motion_channels])
            self.assertEqual(labels['lz_features'].shape.as_list(), [self.batch_size, self.num_unrolls, 19, 19, 18])
            self.assertEqual(labels['targets'].shape.as_list(), [self.batch_size, self.num_unrolls, 19, 19, self.num_target_channels])
            self.assertEqual(labels['targets_mask'].shape.as_list(), [self.batch_size, self.num_unrolls, self.num_target_channels])
            self.assertEqual(labels['value'].shape.as_list(), [self.batch_size, self.num_unrolls, 1])
            self.assertEqual(labels['policy'].shape.as_list(), [self.batch_size, self.num_unrolls, 362])

    def test_data_type(self):
        features, labels = self.dataset.element_spec
        self.assertEqual(features.dtype, tf.float16)
        self.assertEqual(labels['features'].dtype, tf.float16)
        self.assertEqual(labels['motion_features'].dtype, tf.float16)
        self.assertEqual(labels['lz_features'].dtype, tf.float16)
        self.assertEqual(labels['targets'].dtype, tf.float32)
        self.assertEqual(labels['targets_mask'].dtype, tf.float32)
        self.assertEqual(labels['value'].dtype, tf.float32)
        self.assertEqual(labels['policy'].dtype, tf.float32)

class InputFnTrainingModeTest(unittest.TestCase, CommonInputFnTest):
    def setUp(self):
        self.batch_size = 2
        self.num_unrolls = 3
        self.dataset = input_fn(['fixtures/10_games.sgf'], self.batch_size, True, num_unrolls=self.num_unrolls)

class InputFnTestModeTest(unittest.TestCase, CommonInputFnTest):
    def setUp(self):
        self.batch_size = 2
        self.num_unrolls = 3
        self.dataset = input_fn(['fixtures/10_games.sgf'], self.batch_size, False, num_unrolls=self.num_unrolls)


if __name__ == '__main__':
    unittest.main()
