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
from .recompute_grad import recompute_grad


class BottleneckBlock(tf.keras.layers.Layer):
    """ https://arxiv.org/abs/1512.03385v1 """

    def __init__(self, squeeze_factor=0.25):
        super(BottleneckBlock, self).__init__()

        self.squeeze_factor = squeeze_factor

    @property
    def suffix(self):
        return 'bottleneck'

    def as_dict(self, prefix):
        return {
            **self.conv_1.as_dict(f'{prefix}/conv_1'),
            **self.conv_2.as_dict(f'{prefix}/conv_2'),
            **self.conv_3.as_dict(f'{prefix}/conv_3')
        }

    def build(self, input_shape):
        filters = input_shape[-1]
        squeeze_filters = round(self.squeeze_factor * filters)

        self.conv_1 = BatchNormConv2D(filters=squeeze_filters, kernel_size=1)
        self.conv_2 = BatchNormConv2D(filters=squeeze_filters, kernel_size=3)
        self.conv_3 = BatchNormConv2D(filters=filters, kernel_size=1)

    def call(self, x, training=True):
        def _forward(x, is_recomputing=False):
            """ Returns the result of the forward inference pass on `x` """

            # the 1st convolution
            y = self.conv_1(x, training=training, is_recomputing=is_recomputing)
            y = tf.nn.relu(y)

            # the 2st convolution
            y = self.conv_2(y, training=training, is_recomputing=is_recomputing)
            y = tf.nn.relu(y)

            # the 3nd convolution
            y = self.conv_3(y, training=training, is_recomputing=is_recomputing)
            return tf.nn.relu(x + y)

        return recompute_grad(_forward)(x)
