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

from .batch_norm import BatchNormConv2D
from .residual_block import ResidualBlock


class Dynamics(tf.keras.layers.Layer):
    """ The full neural network used to predict the value and policy tensors for
    a mini-batch of board positions. """

    def __init__(
        self,
        *,
        num_blocks,
        num_channels
    ):
        super(Dynamics, self).__init__()

        self.num_blocks = num_blocks
        self.num_channels = num_channels

    def as_dict(self):
        return {}

    @property
    def l2_weights(self):
        return self.trainable_weights

    def build_stem_layer(self, input_shapes, i):
        return ResidualBlock()

    def build(self, input_shapes):
        self.conv_1 = BatchNormConv2D(filters=self.num_channels)
        self.stem = list([
            self.build_stem_layer(input_shapes, i)
            for i in range(self.num_blocks)
        ])

    def call(self, xs, training=True):
        y = tf.concat([
            xs[0],
            tf.reshape(xs[1][:, :361], [-1, 19, 19, 1])
        ], axis=3)

        y = tf.nn.relu(self.conv_1(y, training=training))

        for layer in self.stem:
            y = layer(y, training=training)

        return y
