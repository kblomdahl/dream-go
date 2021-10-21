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

from .policy_head import PolicyHead
from .value_head import ValueHead


class Predictions(tf.keras.layers.Layer):
    """ The full neural network used to predict the value and policy tensors for
    a mini-batch of board positions. """

    def __init__(
        self,
        *,
        num_policy_channels=8,
        num_value_channels=2
    ):
        super(Predictions, self).__init__()

        self.num_policy_channels = num_policy_channels
        self.num_value_channels = num_value_channels

    def as_dict(self):
        return {
            **self.policy_head.as_dict('predictions/policy'),
            **self.value_head.as_dict('predictions/value')
        }

    @property
    def l2_weights(self):
        return self.policy_head.trainable_weights + self.value_head.trainable_weights

    def build(self, input_shapes):
        self.policy_head = PolicyHead(num_samples=self.num_policy_channels)
        self.value_head = ValueHead(num_samples=self.num_value_channels)

    def call(self, x, training=True):
        p = self.policy_head(x, training=training)
        v, vo, vy = self.value_head(x, training=training)

        return v, vy, p, vo, x