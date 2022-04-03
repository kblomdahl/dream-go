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

from ..layers.policy_head import PolicyHead
from ..layers.value_head import ValueHead
from ..layers.batch_norm import BatchNormDense


class Predictor(tf.keras.layers.Layer):
    """ The full neural network used to predict the value and policy tensors for
    a mini-batch of board positions. """

    def __init__(self, *, layers=4, embeddings_size=722, output_shape):
        super(Predictor, self).__init__()

        self.layers = layers
        self.embeddings_size = embeddings_size
        self._output_shape = output_shape

    def as_dict(self):
        return [
            *[layer.as_dict(flat=False) for layer in self.stem],
            {
                't': 'pred',
                'vs': {
                    **self.policy_head.as_dict('policy', flat=True),
                    **self.value_head.as_dict('value', flat=True)
                }
            }
        ]

    def build(self, input_shapes):
        self.policy_head = PolicyHead(output_shape=self._output_shape)
        self.value_head = ValueHead(output_shape=self._output_shape)
        self.stem = list([
            BatchNormDense(out_dims=self.embeddings_size)
            for _ in range(self.layers)
        ])

    def call(self, x, training=True):
        x = tf.reshape(x, [x.shape[0], -1])
        for layer in self.stem:
            x = tf.nn.relu(layer(x, training=training))

        p = self.policy_head(x, training=training)
        v = self.value_head(x, training=training)

        return v, p
