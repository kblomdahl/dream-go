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

from ..layers.batch_norm import BatchNormConv2D
from ..layers.residual_block import ResidualBlock


class TargetPredictorModel(tf.keras.layers.Layer):
    """ The neural network that is responsible for creating the features as
    given by the board state and transforms it into some internal
    representation. """

    def __init__(
        self,
        *,
        num_blocks,
        num_channels,
        num_targets,
        output_shape
    ):
        super(TargetPredictorModel, self).__init__()

        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.num_targets = num_targets
        self._output_shape = output_shape

    def as_dict(self):
        return [
            self.conv_x.as_dict(flat=False),
            *[layer.as_dict() for layer in self.stem],
            self.conv_y.as_dict(flat=False),
        ]

    def build_stem_layer(self, input_shapes, i):
        return ResidualBlock(lambda: BatchNormConv2D(kernel_size=3))

    def build(self, input_shapes):
        self.conv_x = BatchNormConv2D(filters=self.num_channels, kernel_size=3)
        self.conv_y = BatchNormConv2D(filters=self.num_targets, kernel_size=3)
        self.stem = list([
            self.build_stem_layer(input_shapes, i)
            for i in range(self.num_blocks)
        ])

    def call(self, x, training=True):
        y = tf.nn.relu(self.conv_x(x, training=training))

        for layer in self.stem:
            y = layer(y, training=training)

        return tf.reshape(
            tf.nn.relu(self.conv_y(y, training=training)),
            self._output_shape
        )
