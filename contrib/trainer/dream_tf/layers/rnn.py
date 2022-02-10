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

from .batch_norm import XavierOrthogonalInitializer
from .to_dict import tensor_to_dict

class RNN(tf.keras.layers.GRU, XavierOrthogonalInitializer):
    def __init__(self, units):
        super().__init__(
            units=units,
            kernel_initializer=self.xavier_orthogonal_initializer(units, units),
            recurrent_initializer=self.xavier_orthogonal_initializer(units, units),
            return_sequences=True,
            time_major=True
        )

    def as_dict(self, prefix=None, flat=True):
        # This is based on tensorflow [1] internals and is probably subject
        # to change :'(.
        #
        # [1] https://github.com/keras-team/keras/blob/a5a6a53eceb4bb0957fcbe577f56941ae8062d8f/keras/layers/recurrent_v2.py#L629
        units = self.cell.units
        kernel = self.cell.kernel
        recurrent_kernel = self.cell.recurrent_kernel
        offset = self.cell.bias[0, :] # shape is (2, 2166)
        recurrent_offset = self.cell.bias[1, :]

        if flat is True:
            return {
                f'{prefix}/units': tensor_to_dict(units, 'i4'),
                f'{prefix}/reset': tensor_to_dict(tf.transpose(kernel[:, units:units * 2])),
                f'{prefix}/reset/offset': tensor_to_dict(offset[units:units * 2]),
                f'{prefix}/update': tensor_to_dict(tf.transpose(kernel[:, :units])),
                f'{prefix}/update/offset': tensor_to_dict(offset[:units]),
                f'{prefix}/candidate': tensor_to_dict(tf.transpose(kernel[:, units * 2:])),
                f'{prefix}/candidate/offset': tensor_to_dict(offset[units * 2:]),
                f'{prefix}/recurrent_reset': tensor_to_dict(tf.transpose(recurrent_kernel[:, units:units * 2])),
                f'{prefix}/recurrent_reset/offset': tensor_to_dict(recurrent_offset[units:units * 2]),
                f'{prefix}/recurrent_update': tensor_to_dict(tf.transpose(recurrent_kernel[:, :units])),
                f'{prefix}/recurrent_update/offset': tensor_to_dict(recurrent_offset[:units]),
                f'{prefix}/recurrent_candidate': tensor_to_dict(tf.transpose(recurrent_kernel[:, units * 2:])),
                f'{prefix}/recurrent_candidate/offset': tensor_to_dict(recurrent_offset[units * 2:])
            }
        else:
            return [
                {
                    't': 'gru',
                    'vs': {
                        'units': tensor_to_dict(units, 'i4'),
                        'reset': tensor_to_dict(tf.transpose(kernel[:, units:units * 2])),
                        'reset/offset': tensor_to_dict(offset[units:units * 2]),
                        'update': tensor_to_dict(tf.transpose(kernel[:, :units])),
                        'update/offset': tensor_to_dict(offset[:units]),
                        'candidate': tensor_to_dict(tf.transpose(kernel[:, units * 2:])),
                        'candidate/offset': tensor_to_dict(offset[units * 2:]),
                        'recurrent_reset': tensor_to_dict(tf.transpose(recurrent_kernel[:, units:units * 2])),
                        'recurrent_reset/offset': tensor_to_dict(recurrent_offset[units:units * 2]),
                        'recurrent_update': tensor_to_dict(tf.transpose(recurrent_kernel[:, :units])),
                        'recurrent_update/offset': tensor_to_dict(recurrent_offset[:units]),
                        'recurrent_candidate': tensor_to_dict(tf.transpose(recurrent_kernel[:, units * 2:])),
                        'recurrent_candidate/offset': tensor_to_dict(recurrent_offset[units * 2:])
                    }
                }
            ]
