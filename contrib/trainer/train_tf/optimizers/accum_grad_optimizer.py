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


class AccumGradOptimizer(tf.train.Optimizer):
    def __init__(self, _optimizer, num_iters=1):
        super(AccumGradOptimizer, self).__init__(name='AccumGradOptimizer', use_locking=True)

        self._optimizer = _optimizer
        self._num_iters = num_iters

    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads_and_vars = grads_and_vars
        vs = list([v for g, v in grads_and_vars])

        with tf.control_dependencies(None):
            slots = list([self._zeros_slot(v, 'accum', self._name) for v in vs])
            slots_and_vars = list([(s, v) for s, v in zip(slots, vs)])

            # Create the counter on the same device as the first variable.
            with tf.variable_scope(self._name), vs[0].graph.colocate_with(vs[0]):
                counter = tf.Variable(0, trainable=False, dtype=tf.int64)

        with tf.name_scope('AccumGradOptimizer'):
            update_grad_ops = list([s.assign_add(g / tf.cast(self._num_iters, g.dtype)) for s, (g, _v) in zip(slots, grads_and_vars)])
            update_slot_op = tf.group(
                tf.assign_add(counter, 1),
                *update_grad_ops,
            )

            def update_grad():
                update_op = self._optimizer.apply_gradients(slots_and_vars)
                with tf.control_dependencies([update_op]):
                    clear_ops = list([tf.assign(s, tf.zeros_like(s)) for s in slots])
                return tf.group(*clear_ops)

            pred = tf.equal(tf.mod(counter, self._num_iters), 0)
            with tf.control_dependencies([update_slot_op]):
                op = tf.cond(pred, update_grad, tf.no_op)

            if global_step is not None:
                global_step_increment = tf.assign_add(global_step, 1)
                op = tf.group(op, global_step_increment, name=name)
            else:
                op = tf.identity(op, name=name).op

        return op
