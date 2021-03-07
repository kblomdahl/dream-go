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

from ..ffi.libdg_go import get_num_features

""" The total number of input features """
NUM_FEATURES = get_num_features()

""" The type to perform convolution operations in """
COMPUTE_TYPE = tf.float16


def normalize_constraint(x):
    """ Returns a constraint that set each output vector to `tf.norm(x) <= 1` """
    out_dims = x.shape[-1]
    x_f = tf.reshape(x, (-1, out_dims))
    n = tf.norm(x_f, axis=0)
    d = tf.clip_by_value(n, 0.001, tf.math.rsqrt(tf.cast(out_dims, tf.float32)))
    x_n = x_f * d / n

    return tf.reshape(x_n, x.shape)


def unit_constraint(x):
    """ Return a constraint that clip `x` to the range [0, 1] """
    return tf.clip_by_value(x, 0.0, 1.0)


def l2_regularizer(x):
    """ Return the L2 loss of `x` """
    return tf.nn.l2_loss(x)


def conv2d(x, weights):
    """ Shortcut for `tf.nn.conv2d` """
    return tf.nn.conv2d(x, cast_to_compute_type(weights), (1, 1, 1, 1), 'SAME', True, 'NHWC')


def matmul(x, weights, offset=None):
    """ Shortcut for `tf.matmul` """
    y = tf.matmul(x, cast_to_compute_type(weights))

    if offset is not None:
        return y + cast_to_compute_type(offset)
    else:
        return y


def cast_to_compute_type(var):
    """ Returns the given variable as the current compute type """
    return tf.cast(var, COMPUTE_TYPE)


def set_compute_type(type):
    """ Sets the compute type of the convolution operation, and other operations """
    global COMPUTE_TYPE

    COMPUTE_TYPE = type
