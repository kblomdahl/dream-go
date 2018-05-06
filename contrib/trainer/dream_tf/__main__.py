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

import argparse
import base64
import json
import math
from datetime import datetime

import numpy as np
import tensorflow as tf

NUM_FEATURES = 32  # the total number of input features
NUM_BINS = 65536  # the number of bins to use internally when determining scaling

MAX_STEPS = 52428800  # the default total number of examples to train over
BATCH_SIZE = 512  # the default number of examples per batch

LSUV_OPS = 'LSUVOps'  # the graph collection that contains all lsuv operations
DUMP_OPS = 'DumpOps'  # the graph collection that contains all dump operations
HIST_OPS = 'HistOps'  # the graph collection that contains all histogram operations
OUTPUT_OPS = 'OutputOps'  # the graph collection that contains all output tensors
NORM_OPS = 'NormOps'  # the graph collection that contains all weights to be normalized
LEARNING_RATE = 'LearningRate'  # the graph collection that contains the learning rate
LOSS = 'Loss'  # the graph collection that contains the loss


# -------- Initialization --------

def orthogonal_initializer():
    """ Returns an orthogonal initializer that use QR-factorization to find
    the orthogonal basis of a random matrix. This differs from the Tensorflow
    implementation in that it checks for singular matrices, which is a
    problem when generating small matrices. """

    def _init(shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = tf.float32

        assert len(shape) >= 2

        # flatten the input shape with the last dimension remaining so it works
        # for convolutions
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]

        if num_rows < num_cols:
            flat_shape = (num_cols, num_rows)
        else:
            flat_shape = (num_rows, num_cols)

        # keep trying until we encounter a random matrix that is not singular.
        while True:
            a = np.random.standard_normal(flat_shape)
            q, r = np.linalg.qr(a)
            d = np.diag(r)

            if np.prod(d) > 1e-2:
                break

        ph = d / np.abs(d)
        q *= ph

        if num_rows < num_cols:
            q = np.transpose(q, [1, 0])

        return np.reshape(q, shape) / np.linalg.norm(q, ord=2)

    return _init


# -------- Learning Rate Schedule --------

import scipy.stats

class LearningRateScheduler(tf.train.SessionRunHook):
    """
    An automatic learning rate scheduler [1] that decrease the learning rate
    by a factor of 3.0 when the loss reach a plateau.

    [1] Dlib, "Automatic Learning Rate Scheduling That Really Works",
        http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html
    """

    BUF_SIZE = 4096  # the number of samples to calculate the slope over
    THRESHOLD = 2048  # the minimum distance between two decreases in learning rate

    def begin(self):
        self.global_step = tf.train.get_global_step()
        self.learning_rate = tf.get_collection(LEARNING_RATE)[-1]
        self.loss = tf.get_collection(LOSS)[-1]

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
        p = scipy.stats.norm.cdf(0.0, loc=m, scale=math.sqrt(variance))

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

        if steps > LearningRateScheduler.THRESHOLD:
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
            if p < 0.51 and rp < 0.51:
                if learning_rate < 1e-8:
                    run_context.request_stop()

                run_context.session.run([self.learning_rate_op, self.last_decrease_op], feed_dict={
                    self.learning_rate_ph: learning_rate / 3.0,
                    self.last_decrease_ph: global_step
                })


# -------- Graph Components --------

def batch_norm(x, weights, mode, params, suffix=None):
    """ Batch normalization layer. """
    if not suffix:
        suffix = ''

    num_channels = weights.shape[3]
    ones_op = tf.ones_initializer()
    zeros_op = tf.zeros_initializer()

    with tf.variable_scope('batch_norm', reuse=tf.AUTO_REUSE):
        scale = tf.get_variable('scale'+suffix, (num_channels,), tf.float32, ones_op, trainable=False)
        mean = tf.get_variable('mean'+suffix, (num_channels,), tf.float32, zeros_op, trainable=False)
        variance = tf.get_variable('variance'+suffix, (num_channels,), tf.float32, ones_op, trainable=False)

    offset = tf.get_variable('offset'+suffix, (num_channels,), tf.float32, zeros_op, trainable=True)

    # fix the weights so that they appear in the _correct_ order according
    # to cuDNN.
    #
    # tensorflow: [h, w, in, out]
    # cudnn:      [out, in, h, w]
    weights_ = tf.transpose(weights, [3, 2, 0, 1])

    # fold the batch normalization into the convolutional weights and one
    # additional bias term. By scaling the weights and the mean by the
    # term `1 / sqrt(variance + 0.001)`.
    #
    # Also multiply the mean by -1 since the bias term uses addition, while
    # batch normalization assumes subtraction.
    #
    # The weights are scaled using broadcasting, where all input weights for
    # a given output feature are scaled by that features term.
    #
    std_ = tf.sqrt(variance + 0.001)
    offset_ = offset - mean / std_
    weights_ = tf.multiply(
        weights_,
        tf.reshape(scale / std_, (weights_.shape[0], 1, 1, 1))
    )

    tf.add_to_collection(DUMP_OPS, [offset, offset_])
    tf.add_to_collection(DUMP_OPS, [weights, weights_])

    def _forward(x):
        """ Returns the result of the forward inference pass on `x` """
        if mode == tf.estimator.ModeKeys.TRAIN:
            y, b_mean, b_variance = tf.nn.fused_batch_norm(
                x,
                scale,
                offset,
                None,
                None,
                data_format='NCHW',
                is_training=True
            )

            with tf.device(None):
                update_mean_op = tf.assign_sub(mean, 0.01 * (mean - b_mean), use_locking=True)
                update_variance_op = tf.assign_sub(variance, 0.01 * (variance - b_variance), use_locking=True)

                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean_op)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance_op)
        else:
            y, _, _ = tf.nn.fused_batch_norm(
                x,
                scale,
                offset,
                mean,
                variance,
                data_format='NCHW',
                is_training=False
            )

        return y

    return _forward(x)


def residual_block(x, mode, params):
    """
    A single residual block as described by DeepMind.

    1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    5. Batch normalisation
    6. A skip connection that adds the input to the block
    7. A rectifier non-linearity

    We replace step 1 with two parallel convolutions, the first has a dilation
    of one and the second has a dilation of two. We then concatenate the result
    of these convolution before step 2.
    """
    init_op = orthogonal_initializer()
    num_channels = params['num_channels']
    num_dilation = 32 * max(1, math.floor(num_channels / 176.0))
    num_normal = num_channels - num_dilation

    conv_1c = tf.get_variable('weights_1c', (3, 3, num_channels, num_normal), tf.float32, init_op)
    conv_1d = tf.get_variable('weights_1d', (3, 3, num_channels, num_dilation), tf.float32, init_op)
    conv_2 = tf.get_variable('weights_2', (3, 3, num_channels, num_channels), tf.float32, init_op)

    tf.add_to_collection(NORM_OPS, conv_1c)
    tf.add_to_collection(NORM_OPS, conv_1d)
    tf.add_to_collection(NORM_OPS, conv_2)

    def _forward(x):
        """ Returns the result of the forward inference pass on `x` """

        # the 1st convolution
        c = tf.nn.conv2d(x, conv_1c, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        c = batch_norm(c, conv_1c, mode, params, suffix='_1c')
        d = tf.nn.conv2d(x, conv_1d, (1, 1, 1, 1), 'SAME', True, 'NCHW', (1, 1, 2, 2))
        d = batch_norm(d, conv_1d, mode, params, suffix='_1d')

        y = tf.concat([c, d], axis=1)
        y = tf.nn.relu6(y)
        tf.add_to_collection(OUTPUT_OPS, tf.identity(y, 'output_1c'))
        tf.add_to_collection(OUTPUT_OPS, tf.identity(y, 'output_1d'))
        tf.add_to_collection(OUTPUT_OPS, tf.identity(y, 'output_1'))

        # the 2nd convolution
        y = tf.nn.conv2d(y, conv_2, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = batch_norm(y, conv_2, mode, params, suffix='_2')
        y = tf.nn.relu6(y + x)
        tf.add_to_collection(OUTPUT_OPS, tf.identity(y, 'output_2'))

        return y

    return _forward(x)


def value_head(x, mode, params):
    """
    The value head attached after the residual blocks as described by DeepMind:

    1. A convolution of 1 filter of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer to a hidden layer of size 256
    5. A rectifier non-linearity
    6. A fully connected linear layer to a scalar
    7. A tanh non-linearity outputting a scalar in the range [-1, 1]
    """
    init_op = orthogonal_initializer()
    zeros_op = tf.zeros_initializer()
    num_channels = params['num_channels']

    downsample = tf.get_variable('downsample', (1, 1, num_channels, 1), tf.float32, init_op)
    weights_1 = tf.get_variable('weights_1', (361, 256), tf.float32, init_op)
    weights_2 = tf.get_variable('weights_2', (256, 1), tf.float32, init_op)
    bias_1 = tf.get_variable('bias_1', (256,), tf.float32, zeros_op)
    bias_2 = tf.get_variable('bias_2', (1,), tf.float32, zeros_op)

    tf.add_to_collection(NORM_OPS, downsample)

    tf.add_to_collection(DUMP_OPS, [weights_1, weights_1])
    tf.add_to_collection(DUMP_OPS, [weights_2, weights_2])
    tf.add_to_collection(DUMP_OPS, [bias_1, bias_1])
    tf.add_to_collection(DUMP_OPS, [bias_2, bias_2])

    def _forward(x):
        """ Returns the result of the forward inference pass on `x` """
        y = tf.nn.conv2d(x, downsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = batch_norm(y, downsample, mode, params)
        y = tf.nn.relu6(y)
        tf.add_to_collection(OUTPUT_OPS, tf.identity(y, 'output_1'))

        y = tf.reshape(y, (-1, 361))
        y = tf.matmul(y, weights_1) + bias_1
        y = tf.nn.relu(y)
        y = tf.matmul(y, weights_2) + bias_2

        return tf.nn.tanh(y)

    return _forward(x)


def policy_head(x, mode, params):
    """
    The policy head attached after the residual blocks as described by DeepMind:

    1. A convolution of 2 filters of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer that outputs a vector of size 19²+1 = 362
       corresponding to logit probabilities for all intersections and the pass
       move
    """
    init_op = orthogonal_initializer()
    zeros_op = tf.zeros_initializer()
    num_channels = params['num_channels']

    downsample = tf.get_variable('downsample', (1, 1, num_channels, 2), tf.float32, init_op)
    weights = tf.get_variable('weights', (722, 362), tf.float32, init_op)
    bias = tf.get_variable('bias', (362,), tf.float32, zeros_op)

    tf.add_to_collection(NORM_OPS, downsample)

    tf.add_to_collection(DUMP_OPS, [weights, weights])
    tf.add_to_collection(DUMP_OPS, [bias, bias])

    def _forward(x):
        """ Returns the result of the forward inference pass on `x` """
        y = tf.nn.conv2d(x, downsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = batch_norm(y, downsample, mode, params)
        y = tf.nn.relu6(y)
        tf.add_to_collection(OUTPUT_OPS, tf.identity(y, 'output_1'))

        y = tf.reshape(y, (-1, 722))
        y = tf.matmul(y, weights) + bias

        return y

    return _forward(x)

def tower(x, mode, params):
    """ The full neural network used to predict the value and policy tensors for
    a mini-batch of board positions. """
    init_op = orthogonal_initializer()
    num_blocks = params['num_blocks']
    num_channels = params['num_channels']
    num_inputs = NUM_FEATURES

    with tf.variable_scope('01_upsample'):
        upsample = tf.get_variable('weights', (3, 3, num_inputs, num_channels), tf.float32, init_op)
        tf.add_to_collection(NORM_OPS, upsample)

        y = tf.nn.conv2d(x, upsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = batch_norm(y, upsample, mode, params)
        y = tf.nn.relu6(y)

        tf.add_to_collection(OUTPUT_OPS, tf.identity(y, 'output'))

    for i in range(num_blocks):
        with tf.variable_scope('{:02d}_residual'.format(2 + i)):
            y = residual_block(y, mode, params)

    # policy head
    with tf.variable_scope('{:02d}p_policy'.format(2 + num_blocks)):
        p = policy_head(y, mode, params)

    # value head
    with tf.variable_scope('{:02d}v_value'.format(2 + num_blocks)):
        v = value_head(y, mode, params)

    return v, p


# -------- Calibration / Dump functions --------

class DumpHook(tf.train.SessionRunHook):
    """ A hook that prints print all tensors registered in the `DUMP_OPS`
    graph collection to standard output at the end of the session. """

    def __init__(self):
        tf.train.SessionRunHook.__init__(self)

        self.max_bounds = {}
        self.histograms = {}

    def before_run(self, context):
        """ Adds the histogram `HIST_OPS` tensors to the run session. """

        fetches = {}

        for (original, hist_op, max_op) in tf.get_collection(HIST_OPS):
            fetches[original.name] = [hist_op, max_op]

        return tf.train.SessionRunArgs(fetches)

    def after_run(self, context, run_values):
        """ Accumulate the histogram and maximum values into a session wide
        value. """

        for (name, (hist, max_value)) in run_values.results.items():
            self.histograms[name] = self.histograms.get(name, 0) + hist
            self.max_bounds[name] = max(
                self.max_bounds.get(name, -math.inf),
                max_value
            )

    def end(self, session):
        assert self.max_bounds.keys() == self.histograms.keys()

        from scipy import stats
        import sys

        # normalize the histograms (in double precision), we throw away
        # the first and the last elements so that the histogram cover
        # the correct range.
        for name in self.histograms:
            h = self.histograms[name][1:-1].astype('f8')  # cast to double

            self.histograms[name] = h / np.linalg.norm(h, 1)

        # find the optimal scale for each activation by using the jensen shannon
        # divergence. This is very similar to the kullback-leibler divergence
        # used by TensorRT [1] but has less numerical issues.
        #
        # [1] http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        output = {}

        def _quantize(x, num_bins):
            if len(x) == num_bins:
                return np.array(x, copy=True)

            assert len(x) % num_bins == 0

            factor = len(x) // num_bins
            z = np.reshape(x, [num_bins, factor])
            z = np.sum(z, axis=1) / (np.count_nonzero(z, axis=1) + 1e-8)  # avoid (cosmetic) division by zero
            z = np.repeat(z, factor) * (np.asarray(x) > 0)

            return z

        def _entropy_calibrate(original, i):
            reference = np.array(original[:i], copy=True)
            reference[i-1] += np.sum(original[i:])
            candidate = _quantize(original[:i], 128)

            # normalize both distributions so that they are valid
            # probability distributions
            reference = reference / np.linalg.norm(reference, 1)
            candidate = candidate / np.linalg.norm(candidate, 1)

            # avoid log(0) because we do not include outliers in the
            # candidate distribution.
            if candidate[i-1] == 0 and reference[i-1] > 0:
                candidate[i-1] = 1e-4

            def js(p, q):
                """ Jensen Shannon Divergence """
                m = 0.5 * (p + q)

                return 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

            return js(reference, candidate)

        for (name, h) in self.histograms.items():
            best_i = min(
                range(128, NUM_BINS + 128, 128),
                key=lambda i: _entropy_calibrate(h, i)
            )

            scale = (best_i + 0.5) * (6.0 / NUM_BINS)
            scale = np.asarray(scale, 'f2').tostring()

            output[name] = {
                's': base64.b85encode(scale, pad=True).decode('ascii')
            }

            print('.', end='', file=sys.stderr, flush=True)

        # dump the variables to JSON in half precision in order to save disk
        # space.
        for (original, value_op) in tf.get_collection(DUMP_OPS):
            value = session.run(value_op)
            scale = np.linalg.norm(value.flatten(), math.inf).astype('f2').tostring()
            value = value.flatten().astype('f2').tostring()

            output[original.name] = {
                's': base64.b85encode(scale, pad=True).decode('ascii'),
                'v': base64.b85encode(value, pad=True).decode('ascii')
            }

        json.dump(output, sys.stdout, sort_keys=True)


# -------- Input Pipeline --------

def get_dataset(files, batch_size=1, is_training=True):
    """ Returns a tf.DataSet initializable iterator over the given files """

    num_bytes = math.ceil((361 * NUM_FEATURES + 1) / 8.0)

    def _decode_and_split(x):
        return tf.split(tf.decode_raw(x, tf.uint8), (num_bytes, 362))

    def _parse_py(features_and_value, policy):
        feature_size = 361 * NUM_FEATURES
        features_and_value = np.unpackbits(features_and_value)
        features = features_and_value[0:feature_size].astype('f')
        value = np.asarray(1.0 if features_and_value[feature_size] > 0 else -1.0, 'f')
        policy = policy.astype('f') / 255.0

        return features, value, policy

    def _parse(features_and_value, policy):
        return tuple(
            tf.py_func(_parse_py, [features_and_value, policy], [tf.float32, tf.float32, tf.float32])
        )

    def _augment(features, value, policy):
        def _identity(image):
            return tf.identity(image)

        def _flip_lr(image):
            return tf.reverse_v2(image, [2])

        def _flip_ud(image):
            return tf.reverse_v2(image, [1])

        def _transpose_main(image):
            return tf.transpose(image, [0, 2, 1])

        def _transpose_anti(image):
            return tf.reverse_v2(tf.transpose(image, [0, 2, 1]), [1, 2])

        def _rot90(image):
            return tf.transpose(tf.reverse_v2(image, [2]), [0, 2, 1])

        def _rot180(image):
            return tf.reverse_v2(image, [1, 2])

        def _rot270(image):
            return tf.reverse_v2(tf.transpose(image, [0, 2, 1]), [2])

        def _apply_random(random, x):
            return tf.case([
                    (tf.equal(random, 0), lambda: _identity(x)),
                    (tf.equal(random, 1), lambda: _flip_lr(x)),
                    (tf.equal(random, 2), lambda: _flip_ud(x)),
                    (tf.equal(random, 3), lambda: _transpose_main(x)),
                    (tf.equal(random, 4), lambda: _transpose_anti(x)),
                    (tf.equal(random, 5), lambda: _rot90(x)),
                    (tf.equal(random, 6), lambda: _rot180(x)),
                    (tf.equal(random, 7), lambda: _rot270(x)),
                ],
                None,
                exclusive=True
            )

        # apply a random transformation to the input features
        random = tf.random_uniform((), 0, 8, tf.int32)
        features = tf.reshape(features, [NUM_FEATURES, 19, 19])
        features = _apply_random(random, features)

        # transforming the policy is _harder_ since it has that extra pass
        # element at the end, so we temporarily remove it while the tensor gets
        # a random transformation applied
        policy, policy_pass = tf.split(policy, (361, 1))
        policy = tf.reshape(_apply_random(random, tf.reshape(policy, [1, 19, 19])), [361])
        policy = tf.concat([policy, policy_pass], 0)

        value = tf.reshape(value, [1])

        return features, value, policy

    def _fix_shape(features, value, policy):
        features = tf.reshape(features, [NUM_FEATURES, 19, 19])
        value = tf.reshape(value, [1])
        policy = tf.reshape(policy, [362])

        return features, value, policy

    def _shuffle_history(features):
        features = tf.split(features, (5, 6, NUM_FEATURES - 11), 0)

        return tf.concat([
            features[0],
            tf.random_shuffle(features[1]),
            features[2]
        ], axis=0)

    def _fix_history(features, value, policy):
        """ Zeros out the history planes for 25% of the features. """
        HISTORY_MASK = np.asarray([1.0] * NUM_FEATURES, 'f4')
        HISTORY_MASK[5:11] = 0.0

        random = tf.random_uniform((), 0, 100, tf.int32)
        features = tf.case(
            [
                (tf.less(random, 25), lambda: features * tf.reshape(HISTORY_MASK, (-1, 1, 1))),
                (tf.less(random, 35), lambda: _shuffle_history(features)),
            ],
            default=lambda: features
        )

        return features, value, policy

    with tf.device('cpu:0'):
        dataset = tf.data.FixedLengthRecordDataset(files, num_bytes + 362)
        dataset = dataset.map(_decode_and_split)
        dataset = dataset.map(_parse)
        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.map(_augment)
            dataset = dataset.map(_fix_history)
            dataset = dataset.shuffle(393408)
        else:
            dataset = dataset.map(_fix_shape)
        dataset = dataset.batch(batch_size)

        return dataset


def input_fn(files, batch_size, is_training):
    return get_dataset(files, batch_size, is_training).map(
        lambda features, value, policy: (features, {'value': value, 'policy': policy})
    )


# -------- Model function --------

def model_fn(features, labels, mode, params):
    value_hat, policy_hat = tower(features, mode, params)

    if labels:
        # determine the loss for each of the components:
        #
        # - L2 regularization
        # - Value head
        # - Policy head
        #
        loss_value = tf.reduce_mean(tf.squared_difference(
            tf.stop_gradient(labels['value']),
            value_hat
        ))
        loss_policy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(labels['policy']),
            logits=policy_hat
        ))

        loss = loss_policy + loss_value
        tf.add_to_collection(LOSS, loss)

        # set an initial learning rate and then rely on the `LearningRateScheduler`
        # hook to decrease it when the loss plateaus
        global_step = tf.train.get_global_step()
        learning_rate = tf.Variable(params['learning_rate'], False, name='lr')
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        tf.add_to_collection(LEARNING_RATE, learning_rate)

        with tf.control_dependencies(update_ops):
            gradients, variables = zip(*optimizer.compute_gradients(
                loss,
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
                colocate_gradients_with_ops=True
            ))

            apply_op = optimizer.apply_gradients(zip(gradients, variables), global_step)

        with tf.control_dependencies([apply_op]):
            train_ops = [tf.assign(var, var / tf.norm(var)) for var in tf.get_collection(NORM_OPS)]
            train_op = tf.group(train_ops)

        # during training it is very useful to plot the norm of the gradients at
        # each tensor so that we can detect the cause of any exploding gradients
        # or similar issues.
        if mode == tf.estimator.ModeKeys.TRAIN:
            for grad, var in zip(gradients, variables):
                var_name = var.op.name

                tf.summary.scalar('gradients/' + var_name, tf.norm(grad))
                tf.summary.scalar('norms/' + var_name, tf.norm(var))

            tf.summary.scalar('loss/policy', loss_policy)
            tf.summary.scalar('loss/value', loss_value)

            tf.summary.scalar('learning_rate', learning_rate)

        # evalutation metrics such as the accuracy is more human readable than
        # the pure loss function. Even if it is considered bad practice to look
        # at the accuracy instead of the loss.
        policy_hot = tf.argmax(labels['policy'], axis=1)
        policy_1 = tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, 1), tf.float32)
        policy_3 = tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, 3), tf.float32)
        policy_5 = tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, 5), tf.float32)
        value_1 = tf.cast(tf.equal(tf.sign(labels['value']), tf.sign(value_hat)), tf.float32)

        tf.summary.scalar('accuracy/policy_1', tf.reduce_mean(policy_1))
        tf.summary.scalar('accuracy/policy_3', tf.reduce_mean(policy_3))
        tf.summary.scalar('accuracy/policy_5', tf.reduce_mean(policy_5))
        tf.summary.scalar('accuracy/value', tf.reduce_mean(value_1))

        eval_metric_ops = {
            'accuracy/policy_1': tf.metrics.mean(policy_1),
            'accuracy/policy_3': tf.metrics.mean(policy_3),
            'accuracy/policy_5': tf.metrics.mean(policy_5),
            'accuracy/value': tf.metrics.mean(value_1),
            'loss/policy': tf.metrics.mean(loss_policy),
            'loss/value': tf.metrics.mean(loss_value)
        }
    else:
        loss = None
        train_op = None
        eval_metric_ops = {}

    # add a histogram of each activation as well as their maximum value to the
    # set of predictions so that we can quantize the neural network.
    predictions = {
        'value': value_hat,
        'policy': tf.nn.softmax(policy_hat)
    }

    for var in tf.get_collection(OUTPUT_OPS):
        max_op = tf.reduce_max(var)
        hist_op = tf.histogram_fixed_width(
            var,
            [0.0, 6.0],
            nbins=NUM_BINS + 2,
            dtype=tf.int64
        )

        tf.add_to_collection(HIST_OPS, [var, hist_op, max_op])

    # get ride of _worthless_ collections that would just clutter up the
    # saved graph. We do this here to avoid sprinkling a lot of conditions all
    # over the code.
    if mode != tf.estimator.ModeKeys.PREDICT:
        tf.get_default_graph().clear_collection(DUMP_OPS)
        tf.get_default_graph().clear_collection(HIST_OPS)

    # put it all together into a specification
    return tf.estimator.EstimatorSpec(
        mode,
        predictions,
        loss,
        train_op,
        eval_metric_ops
    )


# -------- Estimator --------

def parse_args():
    """ Returns an `argparse` parser for dealing with the command-line
    arguments. """

    parser = argparse.ArgumentParser(
        prog='dream_tf',
        description='Neural network optimizer for Dream Go.'
    )

    parser.add_argument('files', nargs=argparse.REMAINDER, help='The binary features files.')

    opt_group = parser.add_argument_group(title='optional configuration')
    opt_group.add_argument('--warm-start', nargs=1, metavar='M', help='initialize weights from the given model')
    opt_group.add_argument('--batch-size', nargs=1, type=int, metavar='N', help='the number of examples per mini-batch')
    opt_group.add_argument('--steps', nargs=1, type=int, metavar='N', help='the total number of examples to train over')
    opt_group.add_argument('--model', nargs=1, help='the directory that contains the model')
    opt_group.add_argument('--name', nargs=1, help='the name of this session')

    op_group = parser.add_mutually_exclusive_group(required=True)
    op_group.add_argument('--start', action='store_true', help='start training of a new model')
    op_group.add_argument('--resume', action='store_true', help='resume training of an existing model')
    op_group.add_argument('--verify', action='store_true', help='evaluate the accuracy of a model')
    op_group.add_argument('--dump', action='store_true', help='print the weights of a model to standard output')
    op_group.add_argument('--print', action='store_true', help='print the value of the given tensor')

    return parser.parse_args()

def most_recent_model():
    """ Returns the directory in `models/` that is the most recent """
    import os

    all_models = ['models/' + m for m in os.listdir('models/')]

    return max(
        [m for m in all_models if os.path.isdir(m)],
        key=os.path.getmtime
    )

args = parse_args()

model_dir = args.model[0] if args.model else None
if not model_dir:
    if args.start:
        model_dir = 'models/' + datetime.now().strftime('%Y%m%d.%H%M')

        if args.name:
            model_dir += '-' + args.name[0] + '/'
        else:
            model_dir += '/'
    else:
        model_dir = most_recent_model()

params = {
    'steps': args.steps[0] if args.steps else MAX_STEPS,
    'batch_size': args.batch_size[0] if args.batch_size else BATCH_SIZE,
    'learning_rate': 0.001 if args.warm_start else 0.01,

    'num_channels': 128,
    'num_blocks': 9,
}

config = tf.estimator.RunConfig(
    session_config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(
            allow_growth = True
        )
    )
)

if args.warm_start:
    warm_start_from = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=args.warm_start[0],
        vars_to_warm_start='[0-9].*'  # only layers
    )
else:
    warm_start_from = None

nn = tf.estimator.Estimator(
    config=config,
    model_fn=model_fn,
    model_dir=model_dir,
    params=params,
    warm_start_from=warm_start_from
)

if args.start or args.resume:
    nn.train(
        input_fn=lambda: input_fn(args.files, params['batch_size'], True),
        hooks=[LearningRateScheduler()],
        steps=params['steps'] // params['batch_size']
    )
elif args.verify:
    #from tensorflow.python import debug as tf_debug
    #hooks = [tf_debug.LocalCLIDebugHook()]

    # iterate over the entire dataset and collect the metric, which we will
    # then pretty-print as a JSON object to standard output
    results = nn.evaluate(
        input_fn=lambda: input_fn(args.files, params['batch_size'], False),
        steps=params['steps'] // params['batch_size']
    )

    print(json.dumps(
        results,
        default=lambda x: float(x) if x != int(x) else int(x),  # handle `Decimal` types
        sort_keys=True,
        separators=(',', ': '),
        indent=4
    ))
elif args.dump:
    # iterate over the entire dataset and collect the histogram of each
    # activation. We will later use this activation to determine the optimal
    # scale of each activation.
    predictor = nn.predict(
        input_fn=lambda: input_fn(args.files, params['batch_size'], False),
        hooks=[DumpHook()]
    )

    for _ in predictor:
        pass
elif args.print:
    # print the value of the given tensors from the latest checkpoint, or if no
    # tensors are given all available tensors.
    if not args.files:
        print(nn.get_variable_names())
    else:
        for var in args.files:
            value = nn.get_variable_value(var)

            print(var, value.tolist())
