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

    def __init__(self):
        super(ValueHead, self).__init__()

    def as_dict(self, prefix=None, flat=True):
        if flat is True:
            return {
                **self.linear_y.as_dict(f'{prefix}/linear_1', flat=True)
            }
        else:
            return {
                't': 'value',
                'vs': {
                    **self.linear_y.as_dict('linear_1', flat=True)
                }
            }

    def build(self, input_shapes):
        self.linear_y = Dense(1, use_bias=True, bias_initializer=value_offset_op, dtype='float32')

    def call(self, x, training=True):
        return tf.nn.tanh(self.linear_y(x, training=training))

def value_offset_op(shape, dtype=None, partition_info=None):
    return np.array([-0.00502319782])
