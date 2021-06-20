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

from . import normalize_constraint

class BatchNormConv2D(tf.keras.layers.Layer):
    def __init__(self, *, filters=None, kernel_size=3):
        super(BatchNormConv2D, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        if self.filters is None:
            self.filters = input_shape[3]
        in_channels = input_shape[3]

        init_op = tf.keras.initializers.GlorotUniform()
        ones_op = tf.keras.initializers.Ones()
        zeros_op = tf.keras.initializers.Zeros()

        self.filter = self.add_weight('filter', (self.kernel_size, self.kernel_size, in_channels, self.filters), tf.float32, init_op)
        self.offset = self.add_weight('offset', (self.filters,), tf.float32, zeros_op, experimental_autocast=False, trainable=True)
        self.scale = self.add_weight('scale', (self.filters,), tf.float32, ones_op, experimental_autocast=False, trainable=False)
        self.mean = self.add_weight('mean', (self.filters,), tf.float32, zeros_op, experimental_autocast=False, trainable=False)
        self.variance = self.add_weight('variance', (self.filters,), tf.float32, ones_op, experimental_autocast=False, trainable=False)

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
        std_ = tf.sqrt(self.variance + 0.001)
        self.offset_ = self.offset - self.mean / std_
        self.filter_ = tf.multiply(
            self.filter._variable,
            tf.reshape(self.scale / std_, (1, 1, 1, self.filters))
        )

        # fix the weights so that they appear in the _correct_ order according
        # to cuDNN (for NHWC):
        #
        # tensorflow: [h, w, in, out]
        # cudnn:      [out, in, h, w]
        self.filter_ = tf.transpose(self.filter_, perm=[3, 0, 1, 2])

    def call(self, x, training=True, is_recomputing=False):
        """ Returns the result of the forward inference pass on `x` """

        y = tf.nn.conv2d(x, normalize_constraint(self.filter), 1, 'SAME', 'NHWC')

        if training:
            y, b_mean, b_variance = tf.compat.v1.nn.fused_batch_norm(
                y,
                self.scale,
                self.offset,
                None,
                None,
                data_format='NHWC',
                is_training=True
            )

            if not is_recomputing:
                with tf.device(None):
                    update_mean_op = self.mean.assign_sub(0.01 * (self.mean - b_mean), use_locking=True)
                    update_variance_op = self.variance.assign_sub(0.01 * (self.variance - b_variance), use_locking=True)

                    with tf.control_dependencies([update_mean_op, update_variance_op]):
                        y = tf.identity(y)
        else:
            y, _, _ = tf.compat.v1.nn.fused_batch_norm(
                y,
                self.scale,
                self.offset,
                self.mean,
                self.variance,
                data_format='NHWC',
                is_training=False
            )

        return y
