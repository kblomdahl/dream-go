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


class ResidualBlock(tf.keras.layers.Layer):
    """
    A single residual block as described by DeepMind.

    1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    5. Batch normalisation
    6. A skip connection that adds the input to the block
    7. A rectifier non-linearity
    """

    def __init__(self, layer_type):
        super(ResidualBlock, self).__init__()

        self.layer_type = layer_type

    def as_dict(self):
        return {
            't': 'residual_block',
            'vs': {
                **self.conv_1.as_dict('conv_1', flat=True),
                **self.conv_2.as_dict('conv_2', flat=True),
            }
        }

    def build(self, input_shape):
        self.conv_1 = self.layer_type()
        self.conv_2 = self.layer_type()

    def call(self, x, training=True):
        def _forward(x):
            """ Returns the result of the forward inference pass on `x` """

            # the 1st convolution
            y = self.conv_1(x, training=training)
            y = tf.nn.relu(y)

            # the 2nd convolution
            y = self.conv_2(y, training=training)
            return tf.nn.relu(x + y)

        return tf.recompute_grad(_forward)(x)
