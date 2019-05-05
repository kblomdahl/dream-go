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

from . import cast_to_compute_type, normalize_constraint, global_depthwise_conv2d
from ..hooks.dump import DUMP_OPS
from .batch_norm import batch_norm
from .orthogonal_initializer import orthogonal_initializer


def gather_excite(x, mode, params, is_recomputing=False):
    """ Returns the excitement vector for the given input tensor [1].

    [1] _Gather-Excite: Exploiting Feature Context in Convolutional Neural
        Networks_, Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Andrea Vedaldi,
        https://arxiv.org/abs/1810.12348
    """

    init_op = orthogonal_initializer()
    zeros_op = tf.zeros_initializer()
    multiplier = params['gather_excite_multiplier']
    num_channels = params['num_channels']
    num_channels_ = num_channels * multiplier

    conv_1 = tf.get_variable('conv_1', (19, 19, num_channels, multiplier), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)
    linear_1 = tf.get_variable('linear_1', (num_channels_, num_channels_), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)
    linear_2 = tf.get_variable('linear_2', (num_channels_, num_channels), tf.float32, init_op, use_resource=True)
    offset_1 = tf.get_variable('linear_1/offset', (num_channels_,), tf.float32, zeros_op, use_resource=True)
    offset_2 = tf.get_variable('linear_2/offset', (num_channels,), tf.float32, zeros_op, use_resource=True)

    if not is_recomputing:
        tf.add_to_collection(DUMP_OPS, [linear_1, linear_1, 'f2'])
        tf.add_to_collection(DUMP_OPS, [linear_2, linear_2, 'f2'])
        tf.add_to_collection(DUMP_OPS, [offset_1, offset_1, 'f2'])
        tf.add_to_collection(DUMP_OPS, [offset_2, offset_2, 'f2'])

    def _forward(x):
        # gather the input features into a single vector `(n, 1, 1, 128 * m)`
        y = batch_norm(global_depthwise_conv2d(x, conv_1), conv_1, mode, params, is_recomputing=is_recomputing)
        y = tf.nn.relu(y)

        # excite the vector similarly to _squeeze excite_
        y = tf.reshape(y, (-1, num_channels_))
        y = tf.matmul(y, cast_to_compute_type(linear_1)) + cast_to_compute_type(offset_1)
        y = tf.nn.relu(y)

        y = tf.matmul(y, cast_to_compute_type(linear_2)) + cast_to_compute_type(offset_2)
        return tf.nn.sigmoid(tf.reshape(y, (-1, 1, 1, num_channels)))

    return _forward(x)
