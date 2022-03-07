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
from .transition_predictor import TransitionPredictor

class RecurrentModel(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        embeddings_size,
        num_trans_layers
    ):
        super(RecurrentModel, self).__init__()

        self.embeddings_size = embeddings_size
        self.transition_predictor = TransitionPredictor(
            embeddings_size=embeddings_size,
            layers=num_trans_layers
        )
        self.initial_state = self.add_weight('initial_state', (self.embeddings_size,), tf.float32)

    def get_initial_state(self, *, batch_size):
        return tf.repeat([self.initial_state], [batch_size], axis=0)

    def as_dict(self):
        return [
            self.linear_h.as_dict(flat=False),
        ]

    def build(self, input_shapes):
        self.linear_h = BatchNormDense(out_dims=self.embeddings_size)
        self.gru_cell = tf.keras.layers.GRUCell(units=self.embeddings_size)

    def call(self, xs, training=True):
        [h, z, a] = xs

        if h is None:
            batch_size = z.shape[0] if z is not None else a.shape[0]
            h = self.get_initial_state(batch_size=batch_size)

        if a is not None:
            _, h = self.gru_cell(inputs=a, states=h, training=training)

        z_hat = self.transition_predictor(h, training=training)
        if z is None:
            z = z_hat

        hidden_state = tf.nn.relu(
            self.linear_h(tf.concat([h, z], axis=1), training=training)
        )

        return [
            hidden_state,
            z_hat
        ]
