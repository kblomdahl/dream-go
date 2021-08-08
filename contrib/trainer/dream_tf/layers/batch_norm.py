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

from .to_dict import tensor_to_dict

class BatchNormDense(tf.keras.layers.Layer):
    def __init__(self, out_dims=None):
        super(BatchNormDense, self).__init__()

        self.out_dims = out_dims

    def build(self, input_shape):
        in_dims = input_shape[-1]

        init_op = tf.keras.initializers.GlorotUniform()

        self.kernel = self.add_weight('kernel', (in_dims, self.out_dims), tf.float32, init_op, experimental_autocast=False)
        self.kernel_constraint = tf.identity  # tf.keras.constraints.MaxNorm(2.0, axis=0)
        self.batch_norm = tf.keras.layers.BatchNormalization(scale=False)

    def as_dict(self, prefix):
        std_ = tf.sqrt(self.batch_norm.moving_variance + self.batch_norm.epsilon)
        offset_ = self.batch_norm.beta - self.batch_norm.moving_mean / std_
        kernel_ = tf.multiply(
            self.kernel_constraint(self.kernel),
            tf.reshape(1.0 / std_, (1, self.out_dims))
        )

        return {
            f'{prefix}:0': tensor_to_dict(kernel_),
            f'{prefix}/offset:0': tensor_to_dict(offset_)
        }

    def call(self, x, training=True, is_recomputing=False):
        kernel = tf.cast(self.kernel_constraint(self.kernel), x.dtype)
        y = tf.linalg.matmul(x, kernel)

        return self.batch_norm(y, training=training)

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

        self.filter = self.add_weight('filter', (self.kernel_size, self.kernel_size, in_channels, self.filters), tf.float32, init_op, experimental_autocast=False)
        self.filter_constraint = tf.identity #tf.keras.constraints.MinMaxNorm(0.001, 1.0 / sqrt(self.filters), axis=[0, 1, 2])
        self.batch_norm = tf.keras.layers.BatchNormalization(scale=False)

    def as_dict(self, prefix):
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
        std_ = tf.sqrt(self.batch_norm.moving_variance + self.batch_norm.epsilon)
        offset_ = self.batch_norm.beta - self.batch_norm.moving_mean / std_
        filter_ = tf.multiply(
            self.filter_constraint(self.filter),
            tf.reshape(1.0 / std_, (1, 1, 1, self.filters))
        )

        # fix the weights so that they appear in the _correct_ order according
        # to cuDNN (for NHWC):
        #
        # tensorflow: [h, w, in, out]
        # cudnn:      [out, in, h, w]
        filter_ = tf.transpose(filter_, perm=[3, 0, 1, 2])

        return {
            f'{prefix}:0': tensor_to_dict(filter_),
            f'{prefix}/offset:0': tensor_to_dict(offset_),
        }

    def call(self, x, training=True, is_recomputing=False):
        """ Returns the result of the forward inference pass on `x` """

        filter = tf.cast(self.filter_constraint(self.filter), x.dtype)
        y = tf.nn.conv2d(x, filter, 1, 'SAME', 'NHWC')

        return self.batch_norm(y, training=training)
