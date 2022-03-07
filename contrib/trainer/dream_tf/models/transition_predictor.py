# Copyright (c) 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

from ..layers.batch_norm import BatchNormDense
from ..layers.residual_block import ResidualBlock

class TransitionPredictor(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        layers=4,
        embeddings_size
    ):
        super(TransitionPredictor, self).__init__()

        self.layers = layers
        self.embeddings_size = embeddings_size

    def as_dict(self, flat=True):
        return list([
            layer.as_dict(flat=flat)
            for layer in range(self.linear_1)
        ])

    def build(self, input_shapes):
        self.linear_1 = list([
            ResidualBlock(lambda: BatchNormDense(out_dims=self.embeddings_size))
            for _ in range(self.layers)
        ])

    def call(self, x, training=True):
        for linear_1 in self.linear_1:
            x = linear_1(x, training=training)

        return x
