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

def normalize_constraint(x):
    """
    Returns a constraint that set each output vector to `tf.norm(x) <= 1` [1]
    [1] Norm matters: efficient and accurate normalization schemes in deep
        networks, https://papers.nips.cc/paper/2018/file/a0160709701140704575d499c997b6ca-Paper.pdf
    """
    out_dims = x.shape[-1]
    x_f = tf.reshape(x, (-1, out_dims))
    n = tf.norm(tensor=x_f, axis=0)
    d = tf.clip_by_value(n, 0.001, tf.math.rsqrt(tf.cast(out_dims, n.dtype)))
    x_n = x_f * tf.math.divide_no_nan(d, n)

    return tf.reshape(x_n, x.shape)

def normalize_getting(getter, *args, **kwargs):
    return normalize_constraint(getter(*args, **kwargs))
