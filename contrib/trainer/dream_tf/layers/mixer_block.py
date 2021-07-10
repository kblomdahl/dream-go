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
from tensorflow.keras.layers import Reshape

from .batch_norm import BatchNormDense
from .recompute_grad import recompute_grad

class MlpBlock(tf.keras.layers.Layer):
    def __init__(self, mlp_dims):
        super(MlpBlock, self).__init__()

        self.mlp_dims = mlp_dims

    def as_dict(self, prefix):
        return {
            **self.dense_1.as_dict(f'{prefix}/linear_1'),
            **self.dense_2.as_dict(f'{prefix}/linear_2')
        }

    def build(self, input_shapes):
        self.dense_1 = BatchNormDense(self.mlp_dims)
        self.dense_2 = BatchNormDense(input_shapes[-1])

    def call(self, x, training=True, is_recomputing=False):
        y = tf.nn.relu(self.dense_1(x, training=training, is_recomputing=is_recomputing))
        return self.dense_2(y, training=training, is_recomputing=is_recomputing)

class MixerBlock(tf.keras.layers.Layer):
    """ https://arxiv.org/abs/2105.01601 """

    def __init__(self, *, tokens_mlp_dims, channels_mlp_dims):
        super(MixerBlock, self).__init__()

        self.tokens_mlp_dims = tokens_mlp_dims
        self.channels_mlp_dims = channels_mlp_dims

    @property
    def suffix(self):
        return 'mixer'

    def as_dict(self, prefix):
        return {
            **self.mlp_block_1.as_dict(f'{prefix}/mlp_1'),
            **self.mlp_block_2.as_dict(f'{prefix}/mlp_2')
        }

    def build(self, input_shapes):
        self.rearrange = Reshape([input_shapes[-3] * input_shapes[-2], input_shapes[-1]])
        self.mlp_block_1 = MlpBlock(self.tokens_mlp_dims)
        self.mlp_block_2 = MlpBlock(self.channels_mlp_dims)
        self.restore = Reshape(input_shapes[1:])

    def call(self, x, training=True):
        def _forward(x, is_recomputing=False):
            """ Returns the result of the forward inference pass on `x` """

            z = self.rearrange(x)

            y = tf.transpose(z, [0, 2, 1])
            y = self.mlp_block_1(y, training=training, is_recomputing=is_recomputing)
            y = tf.transpose(y, [0, 2, 1])
            z = tf.nn.relu(z + y)
            y = self.mlp_block_2(z, training=training, is_recomputing=is_recomputing)

            return tf.nn.relu(x + self.restore(y))

        return recompute_grad(_forward)(x)
