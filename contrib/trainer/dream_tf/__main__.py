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

import blosc
import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from .learning_rate import LEARNING_RATE, LOSS, LearningRateScheduler
from .orthogonal_initializer import orthogonal_initializer

NUM_FEATURES = 32  # the total number of input features
MAX_STEPS = 524288000  # the default total number of examples to train over
BATCH_SIZE = 1024  # the default number of examples per batch

DUMP_OPS = 'DumpOps'  # the graph collection that contains all dump operations
ADVERSARIAL_OPS = 'AdversarialOps'  # the graph collection that contains all training operations for adversarial networks
ADVERSARIAL_VARS = 'AdversarialVars'  # the graph collection that contains all variables for adversarial networks

# -------- Graph Components --------

def normalize_constraint(x):
    """ Returns a constraint that set `tf.norm(x) = 1` """
    return x / tf.norm(x)

def batch_norm(x, weights, mode, params):
    """ Batch normalization layer. """
    num_channels = weights.shape[3]
    ones_op = tf.ones_initializer()
    zeros_op = tf.zeros_initializer()

    with tf.variable_scope(weights.op.name.split('/')[-1]):
        scale = tf.get_variable('scale', (num_channels,), tf.float32, ones_op, trainable=False)
        mean = tf.get_variable('mean', (num_channels,), tf.float32, zeros_op, trainable=False)
        variance = tf.get_variable('variance', (num_channels,), tf.float32, ones_op, trainable=False)
        offset = tf.get_variable('offset', (num_channels,), tf.float32, zeros_op, trainable=True)

    # fix the weights so that they appear in the _correct_ order according
    # to cuDNN.
    #
    # tensorflow: [h, w, in, out]
    # cudnn:      [out, in, h, w]
    weights_ = tf.reshape(weights, [
        weights.shape[0],
        weights.shape[1],
        -1,  # weights.shape[2] / 4
        4,
        weights.shape[3]
    ])
    weights_ = tf.transpose(weights_, [4, 2, 0, 1, 3])

    # fold the batch normalization into the convolutional weights and one
    # additional bias term. By scaling the weights and the mean by the
    # term `scale / sqrt(variance + 0.001)`.
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
        tf.reshape(scale / std_, (weights_.shape[0], 1, 1, 1, 1))
    )

    # quantize the weights to [-127, +127] and the offset to the same range
    # but as a floating point number as required by cuDNN. Note that this range
    # contains 255 values for the range [-6.0, +6.0] for the offset, so the
    # step size becomes `(12.0 / 255.0)`
    weights_max = tf.reduce_max(tf.abs(weights_))
    weights_q, weights_qmin, weights_qmax = tf.quantize(
        weights_,
        -weights_max,
        weights_max,
        tf.qint8,
        'SCALED',
        'HALF_AWAY_FROM_ZERO'
    )

    step_size = 12.0 / 255.0

    tf.add_to_collection(DUMP_OPS, [offset, offset_ / step_size, 'f4', tf.constant(127.0 * step_size)])
    tf.add_to_collection(DUMP_OPS, [weights, weights_q, 'i1', tf.reduce_max([tf.abs(weights_qmin), tf.abs(weights_qmax)])])

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

    conv_1 = tf.get_variable('conv_1', (3, 3, num_channels, num_channels), tf.float32, init_op, constraint=normalize_constraint)
    conv_2 = tf.get_variable('conv_2', (3, 3, num_channels, num_channels), tf.float32, init_op, constraint=normalize_constraint)

    def _forward(x):
        """ Returns the result of the forward inference pass on `x` """

        # the 1st convolution
        y = tf.nn.conv2d(x, tf.cast(conv_1, tf.float16), (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = batch_norm(y, conv_1, mode, params)
        y = tf.nn.relu6(y)

        # the 2nd convolution
        y = tf.nn.conv2d(y, tf.cast(conv_2, tf.float16), (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = batch_norm(y, conv_2, mode, params)
        y = tf.nn.relu6(y + x)

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

    conv_1 = tf.get_variable('conv_1', (1, 1, num_channels, 1), tf.float32, init_op, constraint=normalize_constraint)
    linear_1 = tf.get_variable('linear_1', (361, 256), tf.float32, init_op)
    linear_2 = tf.get_variable('linear_2', (256, 1), tf.float32, init_op)
    offset_1 = tf.get_variable('linear_1/offset', (256,), tf.float32, zeros_op)
    offset_2 = tf.get_variable('linear_2/offset', (1,), tf.float32, zeros_op)

    tf.add_to_collection(DUMP_OPS, [linear_1, linear_1, 'f4'])
    tf.add_to_collection(DUMP_OPS, [linear_2, linear_2, 'f4'])
    tf.add_to_collection(DUMP_OPS, [offset_1, offset_1, 'f4'])
    tf.add_to_collection(DUMP_OPS, [offset_2, offset_2, 'f4'])

    def _forward(x):
        """ Returns the result of the forward inference pass on `x` """
        y = tf.nn.conv2d(x, tf.cast(conv_1, tf.float16), (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = batch_norm(y, conv_1, mode, params)
        y = tf.nn.relu(y)

        y = tf.reshape(y, (-1, 361))
        y = tf.matmul(y, tf.cast(linear_1, tf.float16)) + tf.cast(offset_1, tf.float16)
        y = tf.nn.relu(y)
        y = tf.matmul(y, tf.cast(linear_2, tf.float16)) + tf.cast(offset_2, tf.float16)

        return tf.cast(tf.nn.tanh(y), tf.float32)

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

    conv_1 = tf.get_variable('conv_1', (1, 1, num_channels, 2), tf.float32, init_op, constraint=normalize_constraint)
    linear_1 = tf.get_variable('linear_1', (722, 362), tf.float32, init_op)
    offset_1 = tf.get_variable('linear_1/offset', (362,), tf.float32, zeros_op)

    tf.add_to_collection(DUMP_OPS, [linear_1, linear_1, 'f4'])
    tf.add_to_collection(DUMP_OPS, [offset_1, offset_1, 'f4'])

    def _forward(x):
        """ Returns the result of the forward inference pass on `x` """
        y = tf.nn.conv2d(x, tf.cast(conv_1, tf.float16), (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = batch_norm(y, conv_1, mode, params)
        y = tf.nn.relu(y)

        y = tf.reshape(y, (-1, 722))
        y = tf.matmul(y, tf.cast(linear_1, tf.float16)) + tf.cast(offset_1, tf.float16)

        return tf.to_float(y)

    return _forward(x)

def tower(x, mode, params):
    """ The full neural network used to predict the value and policy tensors for
    a mini-batch of board positions. """
    init_op = orthogonal_initializer()
    num_blocks = params['num_blocks']
    num_channels = params['num_channels']
    num_inputs = NUM_FEATURES

    with tf.variable_scope('01_upsample', reuse=tf.AUTO_REUSE):
        conv_1 = tf.get_variable('conv_1', (3, 3, num_inputs, num_channels), tf.float32, init_op, constraint=normalize_constraint)

        y = tf.nn.conv2d(x, tf.cast(conv_1, tf.float16), (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = batch_norm(y, conv_1, mode, params)
        y = tf.nn.relu6(y)

    for i in range(num_blocks):
        with tf.variable_scope('{:02d}_residual'.format(2 + i), reuse=tf.AUTO_REUSE):
            y = residual_block(y, mode, params)

    # policy head
    with tf.variable_scope('{:02d}p_policy'.format(2 + num_blocks), reuse=tf.AUTO_REUSE):
        p = policy_head(y, mode, params)

    # value head
    with tf.variable_scope('{:02d}v_value'.format(2 + num_blocks), reuse=tf.AUTO_REUSE):
        v = value_head(y, mode, params)

    return v, p

def adversarial(value_hat, policy_hat):
    """ Returns the probability that the current player is black. """

    zeros_op = tf.zeros_initializer()

    with tf.variable_scope('xx_adversarial', reuse=tf.AUTO_REUSE):
        offset = tf.get_variable('offset', (1,), tf.float32, zeros_op)

        tf.add_to_collection(ADVERSARIAL_VARS, offset)

    return tf.sigmoid(value_hat + offset)

class RunAdversarialOpsHook(tf.train.SessionRunHook):
    """ Runs the adversarial training op `n` number of times for each global
    step. """

    def __init__(self):
        self.train_steps = 3

    def before_run(self, run_context):
        train_ops = tf.get_collection(ADVERSARIAL_OPS)

        if train_ops:
            for _ in range(self.train_steps):
                run_context.session.run(train_ops)

# -------- Calibration / Dump functions --------

class DumpHook(tf.train.SessionRunHook):
    """ A hook that prints all tensors registered in the `DUMP_OPS` graph
    collection to standard output at the end of the session. """

    def end(self, session):
        import sys

        # dump the variables to JSON in `f16` precision in order to save disk
        # space.
        output = {}

        for dump_op in tf.get_collection(DUMP_OPS):
            if len(dump_op) == 4:
                original, value_op, as_type, max_value_op = dump_op
                value, max_value = session.run([value_op, max_value_op])
            else:
                assert len(dump_op) == 3

                original, value_op, as_type = dump_op
                value = session.run(value_op)
                max_value = np.max(np.abs(value))

            max_value = np.asarray(max_value).astype('f4').tostring()
            value = value.astype(as_type).tostring()

            output[original.name] = {
                's': base64.b85encode(max_value, pad=True).decode('ascii'),
                'v': base64.b85encode(value, pad=True).decode('ascii')
            }

        json.dump(output, sys.stdout, sort_keys=True)

# -------- Input Pipeline --------

def get_dataset(files, batch_size=1, is_training=True):
    """ Returns a tf.DataSet initializable iterator over the given files """

    def _blosc_decompress_to_i1(features):
        return np.fromstring(blosc.decompress(features), 'i1')

    def _blosc_decompress_to_u1(features):
        return np.fromstring(blosc.decompress(features), 'u1')

    def _parse(record):
        example = tf.parse_single_example(record, {
            "features": tf.VarLenFeature(tf.string),
            "value": tf.FixedLenFeature([], tf.string),
            "policy": tf.VarLenFeature(tf.string),
        })

        features = tf.decode_raw(tf.sparse_tensor_to_dense(example['features'], default_value=""), tf.uint8)
        policy = tf.decode_raw(tf.sparse_tensor_to_dense(example['policy'], default_value=""), tf.uint8)
        value = tf.decode_raw(example['value'], tf.uint8)

        # decompress, and unzigzag the `features` array
        features = tf.py_func(_blosc_decompress_to_i1, [features], tf.int8, False)
        features = tf.cast(features, tf.float16) / 127.0

        # map `value` to the range [-1, +1] as a floating-point number
        value = 2.0 * tf.cast(value, tf.float32) / 255.0 - 1.0

        # decompress, and map `policy` the the range [0, 1] as a floating-point
        # number
        policy = tf.py_func(_blosc_decompress_to_u1, [policy], [tf.uint8], False)
        policy = tf.cast(policy, tf.float32) / 255.0

        # calculate the `is_black` label based on the input features
        is_black = tf.cast(features[0], tf.float32)

        return features, value, policy, is_black

    def _fix_shape(features, value, policy, is_black):
        features = tf.reshape(features, [NUM_FEATURES, 19, 19])
        value = tf.reshape(value, [1])
        policy = tf.reshape(policy, [362])

        return features, value, policy, is_black

    def _augment(features, value, policy, is_black):
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

        return features, value, policy, is_black

    def _shuffle_history(features):
        features = tf.split(features, (5, 6, NUM_FEATURES - 11), 0)

        return tf.concat([
            features[0],
            tf.random_shuffle(features[1]),
            features[2]
        ], axis=0)

    def _fix_history(features, value, policy, is_black):
        """ Zeros out the history planes for 25% of the features. """
        HISTORY_MASK = np.asarray([1.0] * NUM_FEATURES, 'f2')
        HISTORY_MASK[5:11] = 0.0
        HISTORY_MASK = tf.constant(HISTORY_MASK, tf.float16, (NUM_FEATURES, 1, 1))

        random = tf.random_uniform((), 0, 100, tf.int32)
        features = tf.case(
            [
                (tf.less(random, 10), lambda: features * HISTORY_MASK),
                (tf.less(random, 15), lambda: _shuffle_history(features)),
            ],
            default=lambda: features
        )

        return features, value, policy, is_black

    with tf.device('cpu:0'):
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=2)
        dataset = dataset.map(_parse, num_parallel_calls=4)
        dataset = dataset.map(_fix_shape)
        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.map(_augment, num_parallel_calls=4)
            dataset = dataset.map(_fix_history, num_parallel_calls=4)
            dataset = dataset.shuffle(393408)
        dataset = dataset.batch(batch_size)

        return dataset

def input_fn(files, batch_size, is_training):
    return get_dataset(files, batch_size, is_training).map(
        lambda features, value, policy, is_black: (features, {'value': value, 'policy': policy, 'is_black': is_black})
    )

# -------- Model function --------

def model_fn(features, labels, mode, params):
    value_hat, policy_hat = tower(features, mode, params)

    if labels:
        # determine the loss for each of the components:
        #
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

        # determine the adversarial losses that gets applied to the global
        # step, with no adversarial gradients, and the adversarial steps (with
        # no global gradients).
        #
        # The adversarial loss is a combination of the following losses:
        #
        # - Is Black
        #
        loss_black = tf.reduce_mean(tf.squared_difference(
            tf.stop_gradient(labels['is_black']),
            adversarial(value_hat, policy_hat)
        ))

        loss = loss_policy + loss_value - loss_black
        tf.add_to_collection(LOSS, loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # set an initial learning rate and then rely on the `LearningRateScheduler`
            # hook to decrease it when the loss plateaus
            learning_rate = tf.Variable(params['learning_rate'], False, name='lr')

            tf.add_to_collection(LEARNING_RATE, learning_rate)

            # update the adversarial network first
            adversarial_optimizer = tf.train.AdamOptimizer()
            adversarial_vars = tf.get_collection(ADVERSARIAL_VARS)
            adversarial_train_op = adversarial_optimizer.minimize(
                loss_black,
                var_list=adversarial_vars
            )

            tf.add_to_collection(ADVERSARIAL_OPS, adversarial_train_op)

            # after the adversarial network has been updated, update the main
            # network
            global_step = tf.train.get_global_step()
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                gradients, variables = zip(*optimizer.compute_gradients(
                    loss,
                    var_list=list(set(tf.trainable_variables()) - set(adversarial_vars)),
                    aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
                    colocate_gradients_with_ops=True
                ))

                train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)

            # during training it is very useful to plot the norm of the gradients at
            # each tensor so that we can detect the cause of any exploding gradients
            # or similar issues.
            for grad, var in zip(gradients, variables):
                var_name = var.op.name

                if grad is not None:
                    tf.summary.scalar('gradients/' + var_name, tf.norm(grad))
                tf.summary.scalar('norms/' + var_name, tf.norm(var))

            tf.summary.scalar('learning_rate', learning_rate)
        else:
            train_op = None

        tf.summary.scalar('loss/policy', loss_policy)
        tf.summary.scalar('loss/value', loss_value)
        tf.summary.scalar('loss/is_black', loss_black)

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
            'loss/value': tf.metrics.mean(loss_value),
            'loss/is_black': tf.metrics.mean(loss_black)
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

    # get ride of _worthless_ collections that would just clutter up the
    # saved graph. We do this here to avoid sprinkling a lot of conditions all
    # over the code.
    if mode != tf.estimator.ModeKeys.PREDICT:
        tf.get_default_graph().clear_collection(DUMP_OPS)

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
    opt_group.add_argument('--debug', action='store_true', help='enable command-line debugging')

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
    'learning_rate': 3e-5 if args.warm_start else 0.003,

    'num_channels': 128,
    'num_blocks': 9,
}

config = tf.estimator.RunConfig(
    session_config = tf.ConfigProto(
        graph_options = tf.GraphOptions(
            optimizer_options = tf.OptimizerOptions(
                do_common_subexpression_elimination = not args.debug,
                do_constant_folding = not args.debug,
                do_function_inlining = not args.debug
            )
        ),
        gpu_options = tf.GPUOptions(
            allow_growth = True
        )
    )
)

if args.warm_start:
    warm_start_from = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=args.warm_start[0],
        vars_to_warm_start='[0-9x].*'  # only layers
    )
else:
    warm_start_from = None

hooks = [tf_debug.LocalCLIDebugHook()] if args.debug else []
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
        hooks=hooks + [LearningRateScheduler(), RunAdversarialOpsHook()],
        steps=params['steps'] // params['batch_size']
    )
elif args.verify:
    # iterate over the entire dataset and collect the metric, which we will
    # then pretty-print as a JSON object to standard output
    results = nn.evaluate(
        input_fn=lambda: input_fn(args.files, params['batch_size'], False),
        steps=params['steps'] // params['batch_size'],
        hooks=hooks
    )

    print(json.dumps(
        results,
        default=lambda x: float(x) if x != int(x) else int(x),  # handle `Decimal` types
        sort_keys=True,
        separators=(',', ': '),
        indent=4
    ))
elif args.dump:
    predictor = nn.predict(
        input_fn=lambda: input_fn([], params['batch_size'], False),
        hooks=[DumpHook()]
    )

    for _ in predictor:
        pass
elif args.print:
    # tensors are given then print all available tensors with some statistics.
    if not args.files:
        out = {}

        for var in nn.get_variable_names():
            value = np.asarray(nn.get_variable_value(var))

            out[var] = {
                'mean': float(np.average(value)),
                'std': float(np.std(value))
            }

        print(json.dumps(
            out,
            default=lambda x: float(x) if x != int(x) else int(x),  # handle `Decimal` types
            sort_keys=True,
            separators=(',', ': '),
            indent=4
        ))
    else:
        for var in args.files:
            value = nn.get_variable_value(var)

            print(var, value.tolist())
