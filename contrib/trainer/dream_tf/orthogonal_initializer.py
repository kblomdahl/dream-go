# Copyright (c) 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

import numpy as np
import tensorflow as tf

def orthogonal_initializer():
    """ Returns an orthogonal initializer that use QR-factorization to find
    the orthogonal basis of a random matrix. This differs from the Tensorflow
    implementation in that it checks for singular matrices, which is a
    problem when generating small matrices. """

    def _init(shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = tf.float32

        assert len(shape) >= 2

        # flatten the input shape with the last dimension remaining so it works
        # for convolutions
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]

        if num_rows < num_cols:
            flat_shape = (num_cols, num_rows)
        else:
            flat_shape = (num_rows, num_cols)

        # keep trying until we encounter a random matrix that is not singular.
        while True:
            a = np.random.standard_normal(flat_shape)
            q, r = np.linalg.qr(a)
            d = np.diag(r)

            if np.prod(d) > 1e-2:
                break

        ph = d / np.abs(d)
        q *= ph

        if num_rows < num_cols:
            q = np.transpose(q, [1, 0])

        return np.reshape(q, shape) / np.linalg.norm(q, ord=2)

    return _init
