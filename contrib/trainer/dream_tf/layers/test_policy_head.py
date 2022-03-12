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
import numpy as np

from ..test_common import TestUtils
from .policy_head import PolicyHead, policy_offset_op

class PolicyHeadTest(unittest.TestCase, TestUtils):
    def setUp(self):
        self.batch_size = 2
        self.embeddings_size = 32
        self.x = tf.zeros([self.batch_size, self.embeddings_size], tf.float16)
        self.policy_head = PolicyHead(output_shape=[self.batch_size, 1, -1])

    def test_initializer(self):
        self.assertEqual(
            policy_offset_op([362]).shape,
            (362,)
        )

    def test_shape(self):
        self.assertEqual(
            self.policy_head(self.x, training=True).shape,
            [self.batch_size, 1, 362]
        )

    def test_data_type(self):
        self.assertEqual(
            self.policy_head(self.x, training=True).dtype,
            tf.float32
        )

    def test_fit(self):
        history = self.fit_categorical(
            inputs= \
                np.random.random([1, self.embeddings_size])
                    .repeat(self.batch_size, axis=0),
            outputs=self.policy_head,
            labels=self.create_categorical_labels([1, 1, 362]).repeat(self.batch_size, axis=0),
        )

        self.assertDecreasing(history)

if __name__ == '__main__':
    unittest.main()
