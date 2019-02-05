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

from . import conv2d, normalize_constraint, NUM_FEATURES
from ..hooks.dump import DUMP_OPS
from .batch_norm import batch_norm
from .orthogonal_initializer import orthogonal_initializer
from .policy_head import policy_head
from .residual_block import residual_block
from .value_head import value_head


def tower(x, mode, params):
    """ The full neural network used to predict the value and policy tensors for
    a mini-batch of board positions. """
    init_op = orthogonal_initializer()
    num_blocks = params['num_blocks']
    num_channels = params['num_channels']
    num_inputs = NUM_FEATURES

    # store the number of channels in the JSON output so that we do not have to derive
    # this from the shape later.
    num_channels_ = tf.constant(num_channels, name='num_channels', dtype=tf.int32)

    tf.add_to_collection(DUMP_OPS, [num_channels_, num_channels_, 'i4'])

    with tf.variable_scope('01_upsample', reuse=tf.AUTO_REUSE):
        conv_1 = tf.get_variable('conv_1', (3, 3, num_inputs, num_channels), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)

        y = batch_norm(conv2d(x, conv_1), conv_1, mode, params)
        y = tf.nn.relu(y)

    for i in range(num_blocks):
        with tf.variable_scope('{:02d}_residual'.format(2 + i), reuse=tf.AUTO_REUSE):
            y = residual_block(y, mode, params)

    # policy head
    with tf.variable_scope('{:02d}p_policy'.format(2 + num_blocks), reuse=tf.AUTO_REUSE):
        p = policy_head(y, mode, params)

    # value head
    with tf.variable_scope('{:02d}v_value'.format(2 + num_blocks), reuse=tf.AUTO_REUSE):
        v = value_head(y, mode, params)

    return v, p, y
