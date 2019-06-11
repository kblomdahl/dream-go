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

import numpy as np

from . import _add_layer, _add_variable, _add_constant


def serialize_dense(
        input,
        output,
        kernel=None,
        bias=None,
        gamma=None,
        beta=None,
        epsilon=1e-4,
        mean=None,
        variance=None,
        activation=None
):
    assert bias is None or beta is None, 'Batch normalization cannot be used together with a bias'

    def _dump_dense():
        kernel_ = kernel.eval()

        if beta is not None:
            gamma_ = gamma.eval() if gamma is not None else 1.0
            beta_ = beta.eval()
            mean_ = mean.eval()
            variance_ = variance.eval()

            # fold the batch normalization into the dense weights and one
            # additional bias term. By scaling the weights and the mean by the
            # term `scale / sqrt(variance + 0.001)`.
            #
            # Also multiply the mean by -1 since the bias term uses addition, while
            # batch normalization assumes subtraction.
            #
            # The weights are scaled using broadcasting, where all input weights for
            # a given output feature are scaled by that features term.
            #
            std_ = np.sqrt(variance_ + epsilon)
            bias_ = beta_ - mean_ / std_
            kernel_ = np.multiply(
                kernel_,
                np.reshape(gamma_ / std_, [1, -1])
            )
        else:
            bias_ = bias.eval()

        # we model the matrix multiplication using convolutions in the engine, so we
        # need to transpose the matrix since cuDNN expect it in `KC` format instead of
        # the normal matrix format `CK`:
        #
        # tensorflow: [in, out]
        # cudnn:      [out, in]
        #
        kernel_ = np.transpose(kernel_, [1, 0])

        return {
            "type": "Dense",
            "input": [_add_variable(input)],
            "output": [_add_variable(output)],
            "arguments": {
                "activation": activation,
                "kernel": _add_constant(kernel_),
                "bias": _add_constant(bias_),
            }
        }

    _add_layer(_dump_dense)
