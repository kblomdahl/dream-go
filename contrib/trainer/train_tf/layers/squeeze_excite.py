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

from .conv2d import conv2d
from .global_avg_pool import global_avg_pool
from .multiply import multiply
from .reshape import reshape


def squeeze_excite(x, ratio=8):
    """ Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507 """

    num_channels = x.shape.as_list()[-1]

    y = global_avg_pool(x)
    y = reshape(y, [1, 1, num_channels])
    y = conv2d(y, num_channels // ratio, [1, 1], activation='relu', kernel_initializer='lecun_normal')
    y = conv2d(y, num_channels, [1, 1], activation='sigmoid', kernel_initializer='lecun_normal')

    return multiply([x, y])
