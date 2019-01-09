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

import math

import numpy as np
import scipy.stats
import tensorflow as tf

LEARNING_RATE = 'LearningRate'  # the graph collection that contains the learning rate
LOSS = 'Loss'  # the graph collection that contains the loss

class LearningRateScheduler(tf.train.SessionRunHook):
    """
    An automatic learning rate scheduler [1] that decrease the learning rate
    by a factor of 3.0 when the loss reach a plateau.

    [1] Dlib, "Automatic Learning Rate Scheduling That Really Works",
        http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html
    """

    BUF_SIZE = 4096  # the number of samples to calculate the slope over
    THRESHOLD = 2048  # the minimum distance between two decreases in learning rate

    def __init__(self, steps_to_skip=0):
        self.steps_to_skip = steps_to_skip

    def begin(self):
        self.global_step = tf.train.get_global_step()
        self.learning_rate = tf.get_collection(LEARNING_RATE)[-1]
        self.loss = tf.reduce_mean(tf.get_collection(LOSS))

        with tf.device('cpu:0'):
            buf_size = LearningRateScheduler.BUF_SIZE

            # keep track of the loss in a tensorflow variable so that it can
            # survive a checkpoint
            self.losses = tf.get_variable('learning_rate/losses', (buf_size, 3), tf.float32, trainable=False)
            self.losses_ph = tf.placeholder(tf.float32)
            self.losses_op = self.losses.assign(self.losses_ph)

            # keep track of when we last decreased so we don't do it too often
            self.last_decrease = tf.Variable(0, False, name='learning_rate/last_decrease', dtype=tf.int64)
            self.last_decrease_ph = tf.placeholder(tf.int64)
            self.last_decrease_op = self.last_decrease.assign(self.last_decrease_ph)

            # create some variable purely for the purpose of TensorBoard logging
            # of the slope and the probability that the slope is decreasing
            self.slope = tf.Variable(0.0, False, name='learning_rate/slope')
            self.slope_ph = tf.placeholder(tf.float32)
            self.slope_op = self.slope.assign(self.slope_ph)

            self.p_decreasing = tf.Variable(1.0, False, name='learning_rate/decreasing')
            self.p_decreasing_ph = tf.placeholder(tf.float32)
            self.p_decreasing_op = self.p_decreasing.assign(self.p_decreasing_ph)

        tf.summary.scalar('learning_rate/slope', self.slope)
        tf.summary.scalar('learning_rate/p_decreasing', self.p_decreasing)

        # the operations necessary to decrease the learning rate
        self.learning_rate_ph = tf.placeholder(tf.float32)
        self.learning_rate_op = self.learning_rate.assign(self.learning_rate_ph)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(fetches=[
            self.global_step,
            self.learning_rate,
            self.loss,
            self.losses,
            self.last_decrease
        ])

    def is_decreasing(self, x, y):
        n = x.shape[0]
        if n < 5:
            return 1.0, 0.0

        m, c = np.linalg.lstsq(x, y, rcond=None)[0]

        # estimate the probability that the loss is decreasing based on how
        # well the least squares fit the actual data
        y_hat = m * x[:, 0] + c
        variance = 1.0 / (n - 2.0) * np.sum(np.square(y[:-1] - y_hat[:-1]))
        variance = (12.0 * variance) / (n**3 - n)
        p = scipy.stats.norm.cdf(-5e-6, loc=m, scale=math.sqrt(variance))

        return p, m

    def after_run(self, run_context, run_values):
        global_step, learning_rate, loss, losses, last_decrease = run_values.results

        # add the loss of this step to the global state
        index = global_step % LearningRateScheduler.BUF_SIZE
        losses[index, :] = (global_step, 1.0, loss)

        run_context.session.run(self.losses_op, feed_dict={
            self.losses_ph: losses
        })

        # if we have enough data to estimate the slope of the loss, we do so by
        # fitting a straight line to the function `f(global_step) = loss` using
        # least squares.
        steps = global_step - last_decrease

        if global_step > 0 and global_step % 10 == 0:
            x = losses[:global_step, 0:2]
            y = losses[:global_step, 2].T
            p, m = self.is_decreasing(x, y)

            t = np.percentile(y, 90)
            rx, ry = x[y < t, :], y[y < t]
            rp, _rm = self.is_decreasing(rx, ry)

            run_context.session.run([self.slope_op, self.p_decreasing_op], feed_dict={
                self.slope_ph: m,
                self.p_decreasing_ph: p
            })

            # decrease the learning rate if we are not certain whether the slope
            # is decreasing or not
            can_lower_rate = global_step > self.steps_to_skip and steps > LearningRateScheduler.THRESHOLD
            is_not_decreasing = p < 0.51 and rp < 0.51

            if can_lower_rate and is_not_decreasing:
                if learning_rate < 1e-5:
                    run_context.request_stop()

                run_context.session.run([self.learning_rate_op, self.last_decrease_op], feed_dict={
                    self.learning_rate_ph: learning_rate / 3.0,
                    self.last_decrease_ph: global_step
                })
