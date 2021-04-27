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

from . import conv2d, normalize_getting, cast_to_compute_type
from .recompute_grad import recompute_grad
from .orthogonal_initializer import orthogonal_initializer


def ownership_head(x, mode, params):
    """
    The ownership head attached after the residual blocks:

    1. A convolution of 1 filter of kernel size 1 × 1 with stride 1
    2. A tanh non-linearity outputting a scalar in the range [-1, 1]
    """
    init_op = orthogonal_initializer()
    zeros_op = tf.compat.v1.zeros_initializer()
    num_channels = params['num_channels']

    def _forward(x, is_recomputing=False):
        """ Returns the result of the forward inference pass on `x` """
        conv_1 = tf.compat.v1.get_variable('conv_1', (1, 1, num_channels, 1), tf.float32, init_op, custom_getter=normalize_getting, use_resource=True)
        offset_1 = tf.compat.v1.get_variable('conv_1/offset', (1,), tf.float32, zeros_op, use_resource=True)
        y = conv2d(x, conv_1) + cast_to_compute_type(offset_1)

        y = tf.reshape(y, [-1, 361])
        return tf.cast(tf.nn.tanh(y), tf.float32)

    return recompute_grad(_forward)(x)


def ownership_loss(*, labels=None, logits=None):
    categorical_labels = tf.stack([(1 + labels) / 2, (1 - labels) / 2], axis=2)
    categorical_logits = tf.stack([logits, -logits], axis=2)
    loss = tf.compat.v1.losses.softmax_cross_entropy(
        categorical_labels,
        categorical_logits,
        label_smoothing=0.2,
        reduction=tf.compat.v1.losses.Reduction.NONE
    )

    return tf.reduce_mean(input_tensor=loss, axis=[1], keepdims=True)
