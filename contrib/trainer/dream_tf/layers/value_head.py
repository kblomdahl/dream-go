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
        self.num_additional_tokens = 1

    def as_dict(self, prefix):
        return {
            **self.conv_1.as_dict(f'{prefix}/conv_1'),
            **self.linear_1.as_dict(f'{prefix}/linear_1'),
            **self.linear_2.as_dict(f'{prefix}/linear_2'),
            **self.linear_3.as_dict(f'{prefix}/linear_3')
        }

    def build(self, input_shapes):
        self.conv_1 = BatchNormConv2D(filters=self.num_samples, kernel_size=1)
        self.conv_o = Conv2D(filters=1, kernel_size=1, use_bias=False, dtype='float32', data_format='channels_last')

        self.linear_1 = Dense(361 + self.num_additional_tokens, use_bias=True, dtype='float32')
        self.linear_2 = Dense(1, use_bias=True, dtype='float32')
        self.linear_3 = Dense(1, use_bias=True, bias_initializer=value_offset_op, dtype='float32')

    def call(self, x, training=True):
        y = tf.nn.relu(self.conv_1(x, training=training)) # batch_size, 19, 19, num_samples
        o = tf.nn.tanh(tf.reshape(self.conv_o(y, training=training), [-1, 361]))
        z = tf.cast(tf.reshape(y, [-1, 361, self.num_samples]), tf.float32)

        y = tf.transpose(y, [0, 3, 1, 2]) # batch_size, num_samples, 19, 19
        y = tf.reshape(y, [-1, self.num_samples, 361])  # batch_size, num_samples, 361
        y = tf.nn.relu(self.linear_1(y, training=training)) # batch_size * num_samples, 361 + num_additional_tokens
        y = tf.transpose(y, [0, 2, 1]) # batch_size, 361 + num_additional_tokens, num_samples
        y = tf.nn.relu(self.linear_2(y, training=training)) # batch_size, (361 + num_additional_tokens), 1
        y = tf.squeeze(y, axis=[2]) # batch_size, 361 + num_additional_tokens
        y = tf.nn.tanh(self.linear_3(y, training=training)) # batch_size, 1

        return y, o, z

def value_offset_op(shape, dtype=None, partition_info=None):
    return np.array([-0.00502319782])
