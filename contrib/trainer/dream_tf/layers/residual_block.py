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

from . import conv2d, normalize_constraint, unit_constraint, cast_to_compute_type
from ..hooks.dump import DUMP_OPS
from .batch_norm import batch_norm
from .orthogonal_initializer import orthogonal_initializer
from .recompute_grad import recompute_grad


def residual_block(x, mode, params):
    """
    A single residual block as described by DeepMind.

    1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    5. Batch normalisation
    6. A skip connection that adds the input to the block
    7. A rectifier non-linearity
    """
    init_op = orthogonal_initializer()
    half_op = tf.constant_initializer(0.5)
    num_channels = params['num_channels']

    conv_1 = tf.get_variable('conv_1', (3, 3, num_channels, num_channels), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)
    conv_2 = tf.get_variable('conv_2', (3, 3, num_channels, num_channels), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)
    alpha = tf.get_variable('alpha', (), tf.float32, half_op, constraint=unit_constraint, trainable=True, use_resource=True)

    tf.add_to_collection(DUMP_OPS, [alpha, alpha, 'f4'])

    def _forward(x, is_recomputing=False):
        """ Returns the result of the forward inference pass on `x` """

        # the 1st convolution
        y = batch_norm(conv2d(x, conv_1), conv_1, mode, params, is_recomputing=is_recomputing)
        y = tf.nn.relu(y)

        # the 2nd convolution
        y = batch_norm(conv2d(y, conv_2), conv_2, mode, params, is_recomputing=is_recomputing)
        y = tf.nn.relu(cast_to_compute_type(alpha) * y + cast_to_compute_type(1.0 - alpha) * x)

        return y

    return recompute_grad(_forward)(x)
