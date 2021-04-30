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
from .orthogonal_initializer import orthogonal_initializer

""" The total number of input features """
NUM_FEATURES = get_num_features()

""" The type to perform convolution operations in """
COMPUTE_TYPE = tf.float16


def normalize_constraint(x):
    """
    Returns a constraint that set each output vector to `tf.norm(x) <= 1` [1]

    [1] Norm matters: efficient and accurate normalization schemes in deep
        networks, https://papers.nips.cc/paper/2018/file/a0160709701140704575d499c997b6ca-Paper.pdf
    """
    out_dims = x.shape[-1]
    x_f = tf.reshape(x, (-1, out_dims))
    n = tf.norm(tensor=x_f, axis=0)
    d = tf.clip_by_value(n, 0.001, tf.math.rsqrt(tf.cast(out_dims, tf.float32)))
    x_n = x_f * tf.math.divide_no_nan(d, n)

    return tf.reshape(x_n, x.shape)


def normalize_getting(getter, *args, **kwargs):
    return normalize_constraint(getter(*args, **kwargs))


def conv2d(x, op_name, shape):
    """ Shortcut for `tf.nn.conv2d` """
    weights = tf.compat.v1.get_variable(op_name, shape, tf.float32, orthogonal_initializer(), custom_getter=normalize_getting, use_resource=True)
    offset = tf.compat.v1.get_variable(f'{op_name}/offset', [shape[-1]], tf.float32, tf.compat.v1.zeros_initializer(), use_resource=True)

    return tf.nn.conv2d(input=x, filters=cast_to_compute_type(weights), strides=(1, 1, 1, 1), padding='SAME', data_format='NHWC') + cast_to_compute_type(offset)


def cast_to_compute_type(var):
    """ Returns the given variable as the current compute type """
    return tf.cast(var, COMPUTE_TYPE)


def set_compute_type(type):
    """ Sets the compute type of the convolution operation, and other operations """
    global COMPUTE_TYPE

    COMPUTE_TYPE = type
