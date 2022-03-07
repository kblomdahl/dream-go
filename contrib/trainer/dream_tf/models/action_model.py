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

from ..layers.batch_norm import BatchNormDense


class ActionModel(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        embeddings_size
    ):
        super(ActionModel, self).__init__()

        self.embeddings_size = embeddings_size

    def as_dict(self):
        return [
            self.to_embeddings.as_dict(flat=False)
        ]

    def build(self, input_shapes):
        self.to_embeddings = BatchNormDense(out_dims=self.embeddings_size)

    def call(self, x, training=True):
        y = tf.keras.layers.Flatten()(x)
        return tf.nn.relu(self.to_embeddings(y, training=training))
