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

class Dense(tf.keras.layers.Dense):
    def as_dict(self, prefix=None, flat=True):
        # fix the weights so that they appear in the _correct_ order according
        # to cuDNN when implemented using a 1x1 convolution:
        #
        # tensorflow: [in, out]
        # cudnn:      [out, in]
        kernel = tf.transpose(self.kernel, [1, 0])

        if flat is True:
            return {
                f'{prefix}': tensor_to_dict(kernel),
                f'{prefix}/offset': tensor_to_dict(self.bias),
                f'{prefix}/shape': tensor_to_dict(kernel.shape, as_type='i4')
            }
        else:
            return {
                't': 'dense',
                'vs': {
                    'kernel': tensor_to_dict(kernel),
                    'offset': tensor_to_dict(self.bias),
                    '/shape': tensor_to_dict(kernel.shape, as_type='i4')
                }
            }
