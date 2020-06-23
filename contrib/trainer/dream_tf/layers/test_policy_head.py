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
import numpy as np
import unittest

from .policy_head import policy_head, policy_offset_op
from .test_common import TestUtils

class PolicyHeadTest(unittest.TestCase, TestUtils):
    def setUp(self):
        self.batch_size = 1
        self.num_channels = 128
        self.x = tf.placeholder(tf.float16, [self.batch_size, 19, 19, self.num_channels])
        np.random.seed(12345)
        tf.set_random_seed(67890)

    def tearDown(self):
        tf.reset_default_graph()

    @property
    def params(self):
        return {
            "num_channels": self.num_channels
        }

    def test_initializer(self):
        self.assertEqual(
            policy_offset_op([362]).shape,
            (362,)
        )

    def test_shape(self):
        self.assertEqual(
            policy_head(self.x, tf.estimator.ModeKeys.TRAIN, self.params).shape,
            [self.batch_size, 362]
        )

    def test_data_type(self):
        self.assertEqual(
            policy_head(self.x, tf.estimator.ModeKeys.TRAIN, self.params).dtype,
            tf.float32
        )

    def test_fit(self):
        with tf.device('/cpu:0'):
            logits = policy_head(self.x, tf.estimator.ModeKeys.TRAIN, self.params)
            steps = self.fit_categorical(
                inputs= \
                    np.random.random([1, 19, 19, self.num_channels])
                        .repeat(self.batch_size, axis=0),
                labels=self.create_categorical_labels([1, 362]).repeat(self.batch_size, axis=0),
                logits=logits
            )

            self.assertDecreasing([step['loss'] for step in steps])

if __name__ == '__main__':
    unittest.main()
