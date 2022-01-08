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

from .to_dict import tensor_to_dict

class RNN(tf.keras.layers.GRU):
    def as_dict(self, prefix=None, flat=True):
        if flat is True:
            return {
                f'{prefix}/kernel': tensor_to_dict(self.cell.kernel),
                f'{prefix}/recurrent_kernel': tensor_to_dict(self.cell.recurrent_kernel),
                f'{prefix}/offset': tensor_to_dict(self.cell.bias[0, :]),
                f'{prefix}/recurrent_offset': tensor_to_dict(self.cell.bias[1, :])
            }
        else:
            return {
                't': 'gru',
                'vs': {
                    'kernel': tensor_to_dict(self.cell.kernel),
                    'recurrent_kernel': tensor_to_dict(self.cell.recurrent_kernel),
                    'offset': tensor_to_dict(self.cell.bias[0, :]),
                    'recurrent_offset': tensor_to_dict(self.cell.bias[1, :])
                }
            }
