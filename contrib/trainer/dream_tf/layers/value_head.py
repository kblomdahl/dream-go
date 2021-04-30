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

from . import normalize_getting, conv2d, cast_to_compute_type
from .batch_norm import batch_norm_conv2d
from .dense import dense
from .recompute_grad import recompute_grad
from .orthogonal_initializer import orthogonal_initializer


def value_head(x, mode, params):
    """
    The value head attached after the residual blocks as described by DeepMind:

    1. A convolution of 8 filter of kernel size 3 × 3 with stride 1
    2. Batch normalisation
    3. A relu non-linearity
    4. A fully connected linear layer that outputs a vector of size 1
    5. A tanh non-linearity outputting a scalar in the range [-1, 1]
    """
    num_channels = params['num_channels']
    num_samples = 2

    def _forward(x, is_recomputing=False):
        """ Returns the result of the forward inference pass on `x` """
        y = batch_norm_conv2d(x, 'conv_1', (3, 3, num_channels, num_samples), mode, params, is_recomputing=is_recomputing)
        y = tf.nn.relu(y)

        zo = conv2d(y, 'conv_2', [1, 1, num_samples, 1])
        zo = tf.reshape(zo, [-1, 361])
        zo = tf.nn.tanh(zo)

        z = tf.reshape(y, [-1, 361 * num_samples])
        z = dense(z, 'linear_2', (361 * num_samples, 1), value_offset_op, mode, params, is_recomputing=is_recomputing)
        z = tf.nn.tanh(z)

        return tf.cast(z, tf.float32), tf.cast(zo, tf.float32), tf.cast(tf.reshape(y, [-1, 361, num_samples]), tf.float32)

    return _forward(x)


def value_offset_op(shape, dtype=None, partition_info=None):
    return np.array([-0.00502319782])
