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
import tensorflow.keras.backend as K
import base64
import json

""" All variables that are part of the cuDNN graph """
_VARIABLES = {}

""" All computations in the cuDNN graph """
_LAYERS = []

""" Whether to record the graph. """
_ENABLED = False


def serialize_scope(is_training):
    class SerializeScope:
        def __init__(self, enable):
            self.enable = enable

        def __enter__(self):
            global _ENABLED

            _ENABLED = self.enable

            if _ENABLED:
                _LAYERS.clear()
                _VARIABLES.clear()

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    return SerializeScope(is_training)


def _add_layer(layer):
    if _ENABLED:
        _LAYERS.append(layer)


def _add_variable(variable):
    shape = variable.shape.as_list()

    return _VARIABLES.setdefault(variable.name, {
        "shape": [(s or -1) for s in shape],
        "id": len(_VARIABLES)
    })


def _add_constant(value, shape=None):
    half_value = value.astype('f4')
    byte_value = half_value.tobytes()

    return {
        "shape": shape or value.shape,
        "value": base64.b85encode(byte_value, True).decode('ascii'),
        "mean": float(np.mean(half_value, dtype='f4')),
        "std": float(np.std(half_value, dtype='f4')),
    }


def serialize_to(session, inputs, outputs, stream):
    def strip_dict(d):
        if isinstance(d, dict):
            return {name: strip_dict(value) for name, value in d.items() if value is not None}
        return d

    with session.as_default():
        return json.dump(
            {
                "input": {name: _add_variable(inp) for name, inp in inputs.items()},
                "output": {name: _add_variable(outp) for name, outp in outputs.items()},
                "layers": list([strip_dict(layer()) for layer in _LAYERS]),
            },
            stream,
            allow_nan=False,
            separators=(',', ':'),
            sort_keys=True
        )
