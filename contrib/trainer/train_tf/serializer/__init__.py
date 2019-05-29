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

import tensorflow.keras.backend as K
import base64
import json

""" All variables that are part of the cuDNN graph """
VARIABLES = {}

""" All computations in the cuDNN graph """
LAYERS = []


def _add_layer(layer):
    LAYERS.append(layer)


def _add_variable(variable):
    shape = K.int_shape(variable)

    return VARIABLES.setdefault(variable.name, {
        "shape": [(s or -1) for s in shape],
        "id": len(VARIABLES)
    })


def _add_constant(constant):
    value = K.eval(constant).astype('f2')
    value = value.tobytes()

    return {
        "shape": K.int_shape(constant),
        "value": base64.b85encode(value, True).decode('ascii')
    }


def serialize_to(inputs, outputs, stream):
    def strip_dict(d):
        if isinstance(d, dict):
            return {name: strip_dict(value) for name, value in d.items() if value is not None}
        return d

    return json.dump(
        {
            "input": {name: _add_variable(inp) for name, inp in inputs.items()},
            "output": {name: _add_variable(outp) for name, outp in outputs.items()},
            "layers": [strip_dict(layer()) for layer in LAYERS]
        },
        stream,
        allow_nan=False,
        separators=(',', ':'),
        sort_keys=True
    )