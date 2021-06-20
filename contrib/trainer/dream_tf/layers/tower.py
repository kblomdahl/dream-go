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

from .batch_norm import BatchNormConv2D
from .policy_head import PolicyHead
from .residual_block import ResidualBlock
from .value_head import ValueHead


class Tower(tf.keras.layers.Layer):
    """ The full neural network used to predict the value and policy tensors for
    a mini-batch of board positions. """

    def __init__(
        self,
        *,
        num_blocks,
        num_channels,
        num_policy_channels=8,
        num_value_channels=2
    ):
        super(Tower, self).__init__()

        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.num_policy_channels = num_policy_channels
        self.num_value_channels = num_value_channels

    def as_dict(self):
        out = {
            **self.conv_1.as_dict('01_upsample/conv_1'),
            **self.policy_head.as_dict(f'{self.num_blocks + 2:02}p_policy'),
            **self.value_head.as_dict(f'{self.num_blocks + 2:02}v_value')
        }

        for i, residual_block in enumerate(self.residual_blocks):
            out.update(residual_block.as_dict(f'{i + 2:02}_residual'))

        return out

    @property
    def l2_weights(self):
        out = list(self.conv_1.trainable_weights)
        for residual_block in self.residual_blocks:
            out.extend(residual_block.trainable_weights)

        return out

    def build(self, input_shapes):
        self.num_blocks_ = tf.Variable(self.num_blocks, False, name='num_blocks', dtype=tf.int32)
        self.num_channels_ = tf.Variable(self.num_channels, False, name='num_channels', dtype=tf.int32)
        self.num_samples_ = tf.Variable(self.num_policy_channels, False, name='num_samples', dtype=tf.int32)

        self.conv_1 = BatchNormConv2D(filters=self.num_channels)
        self.residual_blocks = list([
            ResidualBlock()
            for _ in range(self.num_blocks)
        ])

        self.policy_head = PolicyHead(num_samples=self.num_policy_channels)
        self.value_head = ValueHead(num_samples=self.num_value_channels)

    def call(self, x, training=True):
        y = tf.nn.relu(self.conv_1(x, training=training))

        for residual_block in self.residual_blocks:
            y = residual_block(y, training=training)

        p = self.policy_head(y, training=training)
        v, vo, vy = self.value_head(y, training=training)

        return v, vy, p, vo, y
