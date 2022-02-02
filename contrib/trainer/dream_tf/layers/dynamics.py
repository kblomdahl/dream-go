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

from .batch_norm import BatchNormDense, BatchNormConv2D
from .residual_block import ResidualBlock
from .quantize import Quantize


class Dynamics(tf.keras.layers.Layer, Quantize):
    """ The full neural network used to predict the value and policy tensors for
    a mini-batch of board positions. """

    def __init__(
        self,
        *,
        num_blocks,
        num_channels,
        embeddings_size
    ):
        super(Dynamics, self).__init__()

        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.embeddings_size = embeddings_size

    def as_dict(self):
        return [
            self.conv_1.as_dict(flat=False),
            *[layer.as_dict() for layer in self.stem],
            self.to_embeddings.as_dict(flat=False)
        ]

    @property
    def l2_weights(self):
        return self.trainable_weights

    def build_stem_layer(self, input_shapes, i):
        return ResidualBlock()

    def build(self, input_shapes):
        if self.num_blocks > 0:
            self.conv_1 = BatchNormConv2D(filters=self.num_channels, kernel_size=3)
            self.stem = list([
                self.build_stem_layer(input_shapes, i)
                for i in range(self.num_blocks)
            ])

        self.to_embeddings = BatchNormDense(out_dims=self.embeddings_size)

    def call(self, x, training=True):
        if self.num_blocks > 0:
            y = tf.nn.relu(self.conv_1(x, training=training))

            for layer in self.stem:
                y = layer(y, training=training)
        else:
            y = x

        return tf.nn.relu(self.to_embeddings(tf.keras.layers.Flatten()(y), training=training))
