# Copyright (c) 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

def moving_average(x, op_name, mode, decay=0.99975):
    """ Returns a moving average variable of `x` with zero-debias [1]

    [1] Adam - A Method for Stochastic Optimization, Kingma et al, https://arxiv.org/abs/1412.6980
    """
    zeros_op = tf.compat.v1.zeros_initializer()

    if op_name is None:
        op_name = x.name.split(':')[0].split('/')[-1]

    with tf.compat.v1.variable_scope(op_name):
        moving_avg = tf.compat.v1.get_variable('moving_average', x.shape, x.dtype, zeros_op, trainable=False, use_resource=True)
        biased_avg = tf.compat.v1.get_variable('biased_average', x.shape, x.dtype, zeros_op, trainable=False, use_resource=True)
        local_step = tf.compat.v1.get_variable('local_step', (), tf.float32, zeros_op, trainable=False, use_resource=True)

    with tf.compat.v1.name_scope(op_name):
        if mode == tf.estimator.ModeKeys.TRAIN:
            biased_avg_op = tf.compat.v1.assign_sub(biased_avg, (1.0 - decay) * (moving_avg - x), use_locking=True)
            local_step_op = local_step.assign_add(1.0, use_locking=True)
            bias_factor = 1.0 - tf.math.pow(decay, local_step_op)
            update_op = tf.compat.v1.assign(moving_avg, biased_avg_op / tf.cast(bias_factor, moving_avg.dtype), use_locking=True)

            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, update_op)

    return moving_avg
