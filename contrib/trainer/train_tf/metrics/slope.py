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


def avg_slope(x, buf_size=1000):
    num_steps = tf.Variable(0, trainable=False)
    steps = tf.Variable([0.0] * buf_size, trainable=False)
    samples = tf.Variable([0.0] * buf_size, trainable=False)

    current_index = num_steps % buf_size
    update_num_steps = tf.assign_add(num_steps, 1)
    update_steps = tf.scatter_update(steps, current_index, tf.cast(update_num_steps, tf.float32))
    update_samples = tf.scatter_update(samples, current_index, x)

    a = tf.stack([
        tf.constant([1.0] * buf_size, dtype=tf.float32),
        update_steps
    ])
    results = tf.linalg.lstsq(
        tf.transpose(a),
        tf.reshape(update_samples, [-1, 1]),
        fast=False
    )
    offset, slope = tf.split(results, 2)

    return tf.reduce_mean(slope)