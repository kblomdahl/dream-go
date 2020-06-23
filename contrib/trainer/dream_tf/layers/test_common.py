# Copyright (c) 2020 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
import numpy as np

class TestUtils(object):
    def create_categorical_labels(self, shape):
        cand = np.random.random(shape)
        return cand / np.sum(cand)

    def fit_categorical(self, inputs, labels, logits, step_ops=None, iter=100):
        labels_ph = tf.placeholder(tf.float32, labels.shape)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels_ph,
                logits=logits
            )
        )

        return self.fit(
            loss=loss,
            inputs=inputs,
            labels=labels,
            labels_ph=labels_ph,
            step_ops=step_ops,
            iter=iter
        )

    def fit_regression(self, inputs, labels, logits, step_ops=None, iter=100):
        labels_ph = tf.placeholder(tf.float32, labels.shape)
        loss = tf.reduce_mean(
            tf.math.squared_difference(logits, labels_ph)
        )

        return self.fit(
            loss=loss,
            inputs=inputs,
            labels=labels,
            labels_ph=labels_ph,
            step_ops=step_ops,
            iter=iter
        )

    def fit(self, loss, inputs, labels, labels_ph, step_ops=None, iter=100):
        optimizer = tf.train.GradientDescentOptimizer(1e-4)

        if step_ops is None:
            step_ops = {'loss': loss}

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            minimize_op = optimizer.minimize(loss)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            return \
                [
                    session.run(
                        [step_ops, minimize_op],
                        feed_dict={
                            self.x: inputs,
                            labels_ph: labels
                        }
                    )[0]
                    for _ in range(iter)
                ]

    def assertDecreasing(self, steps, threshold=-1e-8):
        p = \
            np.polyfit(
                np.arange(len(steps)),
                steps,
                1
            )

        self.assertLess(p[0], threshold)