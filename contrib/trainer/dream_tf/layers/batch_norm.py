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

from ..hooks.dump import DUMP_OPS


def batch_norm(x, weights, mode, params, is_recomputing=False):
    """ Batch normalization layer. """
    num_channels = x.shape[3]
    ones_op = tf.ones_initializer()
    zeros_op = tf.zeros_initializer()

    with tf.variable_scope(weights.op.name.split('/')[-1]):
        scale = tf.get_variable('scale', (num_channels,), tf.float32, ones_op, trainable=True, use_resource=True)
        mean = tf.get_variable('mean', (num_channels,), tf.float32, zeros_op, trainable=False, use_resource=True)
        variance = tf.get_variable('variance', (num_channels,), tf.float32, ones_op, trainable=False, use_resource=True)
        offset = tf.get_variable('offset', (num_channels,), tf.float32, zeros_op, trainable=True, use_resource=True)

    if not is_recomputing:
        is_depthwise = 'depthwise' in x.name

        # fold the batch normalization into the convolutional weights and one
        # additional bias term. By scaling the weights and the mean by the
        # term `scale / sqrt(variance + 0.001)`.
        #
        # Also multiply the mean by -1 since the bias term uses addition, while
        # batch normalization assumes subtraction.
        #
        # The weights are scaled using broadcasting, where all input weights for
        # a given output feature are scaled by that features term.
        #
        std_ = tf.sqrt(variance + 0.001)
        offset_ = offset - mean / std_

        if is_depthwise:
            num_in_channels = weights.shape[2]
            num_in_multiplier = weights.shape[3]

            weights_ = tf.multiply(
                weights,
                tf.reshape(scale / std_, (1, 1, num_in_channels, num_in_multiplier))
            )
        else:
            weights_ = tf.multiply(
                weights,
                tf.reshape(scale / std_, (1, 1, 1, num_channels))
            )

        # fix the weights so that they appear in the _correct_ order according
        # to cuDNN (for NHWC):
        #
        # tensorflow: [h, w, in, out]
        # cudnn:      [out, h, w, in]
        weights_ = tf.transpose(weights_, [3, 0, 1, 2])

        tf.add_to_collection(DUMP_OPS, [offset, offset_, 'f2'])
        tf.add_to_collection(DUMP_OPS, [weights, weights_, 'f2'])

    def _forward(x):
        """ Returns the result of the forward inference pass on `x` """
        if mode == tf.estimator.ModeKeys.TRAIN:
            y, b_mean, b_variance = tf.nn.fused_batch_norm(
                x,
                scale,
                offset,
                None,
                None,
                data_format='NHWC',
                is_training=True
            )

            if not is_recomputing:
                with tf.device(None):
                    update_mean_op = tf.assign_sub(mean, 0.01 * (mean - b_mean), use_locking=True)
                    update_variance_op = tf.assign_sub(variance, 0.01 * (variance - b_variance), use_locking=True)

                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean_op)
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance_op)
        else:
            y, _, _ = tf.nn.fused_batch_norm(
                x,
                scale,
                offset,
                mean,
                variance,
                data_format='NHWC',
                is_training=False
            )

        return y

    return _forward(x)
