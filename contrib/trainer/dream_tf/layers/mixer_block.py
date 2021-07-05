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
from tensorflow.keras.layers import Dense, LayerNormalization, Reshape

from .to_dict import tensor_to_dict
from .recompute_grad import recompute_grad

class MlpBlock(tf.keras.layers.Layer):
    def __init__(self, mlp_dims):
        super(MlpBlock, self).__init__()

        self.mlp_dims = mlp_dims

    def as_dict(self, prefix):
        return {
            f'{prefix}/linear_1:0': tensor_to_dict(self.dense_1.kernel),
            f'{prefix}/linear_1/offset:0': tensor_to_dict(self.dense_1.bias),
            f'{prefix}/linear_2:0': tensor_to_dict(self.dense_2.kernel),
            f'{prefix}/linear_2/offset:0': tensor_to_dict(self.dense_2.bias),
        }

    def build(self, input_shapes):
        self.dense_1 = Dense(self.mlp_dims, activation='relu')
        self.dense_2 = Dense(input_shapes[-1], activation=None)

    def call(self, x):
        y = self.dense_1(x)
        return self.dense_2(y)

class MixerBlock(tf.keras.layers.Layer):
    def __init__(self, *, tokens_mlp_dims, channels_mlp_dims):
        super(MixerBlock, self).__init__()

        self.tokens_mlp_dims = tokens_mlp_dims
        self.channels_mlp_dims = channels_mlp_dims

    def as_dict(self, prefix):
        return {
            # pass
        }

    def build(self, input_shapes):
        self.rearrange = Reshape([input_shapes[-3] * input_shapes[-2], input_shapes[-1]])
        self.layer_norm_1 = LayerNormalization(center=False, scale=False)
        self.mlp_block_1 = MlpBlock(self.tokens_mlp_dims)
        self.layer_norm_2 = LayerNormalization(center=False, scale=False)
        self.mlp_block_2 = MlpBlock(self.channels_mlp_dims)
        self.restore = Reshape(input_shapes[1:])

    def call(self, x, training=True):
        def _forward(x, is_recomputing=False):
            """ Returns the result of the forward inference pass on `x` """

            z = self.rearrange(x)

            y = self.layer_norm_1(z, training=training)
            y = tf.transpose(y, [0, 2, 1])
            y = self.mlp_block_1(y)
            y = tf.transpose(y, [0, 2, 1])
            z = tf.nn.relu(z + y)
            y = self.layer_norm_2(z, training=training)
            y = self.mlp_block_2(y)

            return tf.nn.relu(x + self.restore(y))

        return recompute_grad(_forward)(x)
