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

import base64
import json
import numpy as np
import sys
import tensorflow as tf

""" The graph collection that contains all dump operations """
DUMP_OPS = 'DumpOps'

""" The graph collection that contains all dump string operations """
DUMP_STR_OPS = 'DumpStrOps'


class DumpHook(tf.estimator.SessionRunHook):
    """ A hook that prints all tensors registered in the `DUMP_OPS` graph
    collection to standard output at the end of the session. """

    def end(self, session):
        # dump the variables to JSON in `f16` precision in order to save disk
        # space.
        output = {}

        for dump_op in tf.compat.v1.get_collection(DUMP_OPS):
            if len(dump_op) == 4:
                name, value_op, as_type, max_value_op = dump_op
                value, max_value = session.run([value_op, max_value_op])
            else:
                assert len(dump_op) == 3

                name, value_op, as_type = dump_op
                value = session.run(value_op)
                max_value = np.max(np.abs(value))

            max_value = np.asarray(max_value).astype('f4').tostring()
            value = value.astype(as_type).tostring()

            output[name] = {
                's': base64.b85encode(max_value, pad=True).decode('ascii'),
                't': as_type,
                'v': base64.b85encode(value, pad=True).decode('ascii'),
            }

            for dump_op in tf.compat.v1.get_collection(DUMP_STR_OPS):
                name, value_op = dump_op
                output[name] = session.run(value_op).decode('ascii')

        json.dump(output, sys.stdout, sort_keys=True)
