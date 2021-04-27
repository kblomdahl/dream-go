# Copyright (c) 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

from ..hooks.dump import DUMP_OPS
from .moving_average import moving_average
from .orthogonal_initializer import orthogonal_initializer
from . import cast_to_compute_type

def dense(x, op_name, shape, offset_init_op, mode, params, is_recomputing=False):
    if offset_init_op is None:
        offset_init_op = tf.compat.v1.zeros_initializer()

    weights = tf.compat.v1.get_variable(op_name, shape, tf.float32, orthogonal_initializer(), collections=[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, tf.compat.v1.GraphKeys.WEIGHTS], use_resource=True)
    offset = tf.compat.v1.get_variable(op_name + '/offset', (shape[-1],), tf.float32, offset_init_op, use_resource=True)

    if not is_recomputing and 'no_dump' not in params:
        tf.compat.v1.add_to_collection(DUMP_OPS, [weights.name, moving_average(weights, f'{op_name}/moving_avg', mode), 'f2'])
        tf.compat.v1.add_to_collection(DUMP_OPS, [offset.name, moving_average(offset, f'{op_name}/offset/moving_avg', mode), 'f2'])

    return tf.matmul(x, cast_to_compute_type(weights)) + cast_to_compute_type(offset)
