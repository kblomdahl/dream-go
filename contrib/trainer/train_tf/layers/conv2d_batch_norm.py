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

from ..serializer.conv2d import serialize_conv2d


def conv2d_batch_norm(x, num_channels, filter_size, activation='linear', training=None):
    conv2d = tf.keras.layers.Conv2D(
        num_channels,
        filter_size,
        padding='same',
        use_bias=False,
        kernel_initializer='orthogonal'
    )
    batch_norm = tf.keras.layers.BatchNormalization(renorm=True)

    # forward pass
    y = batch_norm(conv2d(x), training=training)

    if activation != 'linear':
        act = tf.keras.layers.Activation(activation)
        y = act(y)

    # serialize
    serialize_conv2d(
        input=x,
        output=y,

        kernel=conv2d.kernel,
        bias=conv2d.bias if conv2d.use_bias else None,

        gamma=batch_norm.gamma,
        beta=batch_norm.beta,
        epsilon=batch_norm.epsilon,
        mean=batch_norm.moving_mean,
        variance=batch_norm.moving_variance,

        activation=None if activation == 'linear' else activation,
    )

    return y