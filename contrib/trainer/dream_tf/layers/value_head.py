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

import tensorflow as tf

from . import conv2d, normalize_constraint, COMPUTE_TYPE
from ..hooks.dump import DUMP_OPS
from .batch_norm import batch_norm
from .recompute_grad import recompute_grad
from .orthogonal_initializer import orthogonal_initializer


def value_head(x, mode, params):
    """
    The value head attached after the residual blocks as described by DeepMind:

    1. A convolution of 1 filter of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer to a hidden layer of size 256
    5. A rectifier non-linearity
    6. A fully connected linear layer to a scalar
    7. A tanh non-linearity outputting a scalar in the range [-1, 1]
    """
    init_op = orthogonal_initializer()
    zeros_op = tf.zeros_initializer()
    num_channels = params['num_channels']

    conv_1 = tf.get_variable('conv_1', (1, 1, num_channels, 1), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)
    linear_1 = tf.get_variable('linear_1', (361, 256), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)
    linear_2 = tf.get_variable('linear_2', (256, 1), tf.float32, init_op, use_resource=True)
    offset_1 = tf.get_variable('linear_1/offset', (256,), tf.float32, zeros_op, use_resource=True)
    offset_2 = tf.get_variable('linear_2/offset', (1,), tf.float32, zeros_op, use_resource=True)

    tf.add_to_collection(DUMP_OPS, [linear_1, linear_1, 'f2'])
    tf.add_to_collection(DUMP_OPS, [linear_2, linear_2, 'f2'])
    tf.add_to_collection(DUMP_OPS, [offset_1, offset_1, 'f2'])
    tf.add_to_collection(DUMP_OPS, [offset_2, offset_2, 'f2'])

    def _forward(x, is_recomputing=False):
        """ Returns the result of the forward inference pass on `x` """
        y = batch_norm(conv2d(x, conv_1), conv_1, mode, params, is_recomputing=is_recomputing)
        y = tf.nn.relu(y)

        y = tf.reshape(y, (-1, 361))
        y = tf.matmul(y, tf.cast(linear_1, COMPUTE_TYPE)) + tf.cast(offset_1, COMPUTE_TYPE)
        y = tf.nn.relu(y)
        y = tf.matmul(y, tf.cast(linear_2, COMPUTE_TYPE)) + tf.cast(offset_2, COMPUTE_TYPE)

        return tf.cast(tf.nn.tanh(y), tf.float32)

    return recompute_grad(_forward)(x)
