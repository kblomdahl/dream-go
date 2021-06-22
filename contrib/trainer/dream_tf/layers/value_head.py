# Copyright (c) 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from .batch_norm import BatchNormConv2D
from .dense import Dense

class ValueHead(tf.keras.layers.Layer):
    """
    The value head attached after the residual blocks as described by DeepMind:

    1. A convolution of 8 filter of kernel size 3 × 3 with stride 1
    2. Batch normalisation
    3. A relu non-linearity
    4. A fully connected linear layer that outputs a vector of size 1
    5. A tanh non-linearity outputting a scalar in the range [-1, 1]
    """

    def __init__(self, *, num_samples):
        super(ValueHead, self).__init__()

        self.num_samples = num_samples

    def as_dict(self, prefix):
        return {
            **self.conv_1.as_dict(f'{prefix}/conv_1'),
            **self.linear_2.as_dict(f'{prefix}/linear_2')
        }

    def build(self, input_shapes):
        self.conv_1 = BatchNormConv2D(filters=self.num_samples, kernel_size=3)
        self.conv_2 = Conv2D(filters=1, kernel_size=1, use_bias=False, data_format='channels_last')
        self.linear_2 = Dense(1, use_bias=True, bias_initializer=value_offset_op)

    def call(self, x, training=True):
        y = tf.nn.relu(self.conv_1(x, training=training))

        zo = tf.nn.tanh(self.conv_2(y, training=training))
        zo = tf.reshape(zo, [-1, 361])

        y = tf.reshape(y, [-1, 361, self.num_samples])
        z = tf.reshape(y, [-1, 361 * self.num_samples])
        z = tf.nn.tanh(self.linear_2(z, training=training))

        return tf.cast(z, tf.float32), tf.cast(zo, tf.float32), tf.cast(y, tf.float32)


def value_offset_op(shape, dtype=None, partition_info=None):
    return np.array([-0.00502319782])
