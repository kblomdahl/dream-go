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
import tensorflow.keras.backend as K


class AccumulatedNadam(tf.keras.optimizers.Nadam):
    def __init__(self, update_freq=8, **kwargs):
        super(AccumulatedNadam, self).__init__(**kwargs)
        self.update_freq = update_freq

        self.accumulated_gradients = None
        self.accumulated_iterations = K.variable(0, dtype='int64', name='accumulated_iterations')

    def get_gradients(self, loss, params):
        if self.accumulated_gradients is None:
            return [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            return self.accumulated_gradients

    def get_updates(self, loss, params):
        # initialize the accumulators here, since we do not know their shape in
        # the constructor
        if self.accumulated_gradients is None:
            self.accumulated_gradients = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        #
        mini_gradients = super(AccumulatedNadam, self).get_gradients(loss, params)

        update_freq = self.update_freq
        update_accumulated_iterations = K.update_add(self.accumulated_iterations, 1)
        update_accumulated_gradients = [
            K.update_add(acc, g / K.cast(update_freq, K.dtype(g))) for acc, g in zip(self.accumulated_gradients, mini_gradients)
        ]

        def _accumulate():
            """ Increment the counter for how many remaining iterations until we
            should apply the accumulated gradients """
            return tf.group(update_accumulated_iterations)

        def _update():
            """ Apply, and then clear, the accumulated gradients """
            original_updates = super(AccumulatedNadam, self).get_updates(loss, params)
            original_updates = flatten(original_updates)

            with tf.control_dependencies(original_updates):
                clear_accumulated_gradients = [
                    K.update(acc, K.zeros(K.int_shape(acc), dtype=K.dtype(acc))) for acc in self.accumulated_gradients
                ]

            return tf.group(clear_accumulated_gradients, update_accumulated_iterations)

        update_switch = K.equal(update_accumulated_iterations % update_freq, 0)

        with tf.control_dependencies(update_accumulated_gradients):
            return [K.switch(update_switch, _update, _accumulate)]

    def get_config(self):
        config = super(AccumulatedNadam, self).get_config()
        config.update({
            'update_freq': self.update_freq,
        })
        return config


def flatten(l):
    for elem in l:
        if isinstance(elem, list):
            yield from flatten(elem)
        if isinstance(elem, tuple):
            yield from flatten(elem)
        else:
            yield elem
