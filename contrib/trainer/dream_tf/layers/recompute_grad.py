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


def recompute_grad(func):
    @tf.custom_gradient
    def _forward(x):
        # calculate the forward pass, while keeping track of which variables
        # were accessed
        with tf.GradientTape() as tape:
            y = func(x, is_recomputing=False)

        original_vars = set(tape.watched_variables())
        var_scope = tf.compat.v1.get_variable_scope()

        def grad_fn(output_grad, variables=None):
            assert set(variables) == original_vars, 'Wrong variables passed to @tf.custom_gradient function'

            # re-calculate the gradient
            in_var = tf.identity(x)

            with tf.control_dependencies([output_grad]):
                with tf.compat.v1.variable_scope(var_scope, reuse=True):
                    y = func(in_var, is_recomputing=True)

            grads = tf.gradients(
                ys=[y],
                xs=[in_var] + variables,
                grad_ys=[output_grad],
                gate_gradients=True,
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

            return grads[:1], grads[1:]

        return y, grad_fn

    return _forward
