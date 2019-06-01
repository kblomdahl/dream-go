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

from .add import add
from .conv2d_batch_norm import conv2d_batch_norm
from .depthwise_conv2d_batch_norm import depthwise_conv2d_batch_norm
from .squeeze_excite import squeeze_excite


def mb_conv_block(x, expand_ratio=6, training=None):
    """ Mobile Inverted Residual Bottleneck, https://arxiv.org/pdf/1905.11946.pdf """

    num_channels = x.shape.as_list()[-1]
    num_expanded = expand_ratio * num_channels

    # expand
    y = conv2d_batch_norm(x, num_expanded, [1, 1], activation='relu', training=training)

    # mix
    y = depthwise_conv2d_batch_norm(y, [3, 3], activation='relu', training=training)
    y = squeeze_excite(y, training=training)

    # project
    y = conv2d_batch_norm(y, num_channels, [1, 1], activation='linear', training=training)

    return add([x, y])