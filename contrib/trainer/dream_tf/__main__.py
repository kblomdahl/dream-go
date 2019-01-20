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
from datetime import datetime

import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from .boost_table import BOOST_PER_MOVE_NUMBER
from .libgo import get_num_features, get_single_example
from .learning_rate import LEARNING_RATE, LOSS, LearningRateScheduler
from .orthogonal_initializer import orthogonal_initializer, orthogonal_loss
from .pretty_print import to_sgf_heatmap

NUM_FEATURES = get_num_features()  # the total number of input features
MAX_STEPS = 524288000  # the default total number of examples to train over
BATCH_SIZE = 2048  # the default number of examples per batch

DUMP_OPS = 'DumpOps'  # the graph collection that contains all dump operations

# -------- Graph Components --------

def recompute_grad(func):
    @tf.custom_gradient
    def _forward(x):
        # calculate the forward pass, while keeping track of which variables
        # were accessed
        with tf.GradientTape() as tape:
            y = func(x, is_recomputing=False)

        original_vars = set(tape.watched_variables())
        var_scope = tf.get_variable_scope()

        def grad_fn(output_grad, variables=None):
            assert set(variables) == original_vars, "Wrong variables passed to @tf.custom_gradient function"

            # re-calculate the gradient
            in_var = tf.identity(x)

            with tf.control_dependencies([output_grad]):
                with tf.variable_scope(var_scope, reuse=True):
                    y = func(in_var, is_recomputing=True)

            grads = tf.gradients(
                [y],
                [in_var] + variables,
                [output_grad],
                gate_gradients=True,
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
                colocate_gradients_with_ops=True
            )

            return grads[:1], grads[1:]

        return y, grad_fn

    return _forward

def normalize_constraint(x):
    """ Returns a constraint that set each output vector to `tf.norm(x) = 1` """
    out_dims = x.shape[-1]
    x_f = tf.reshape(x, (-1, out_dims))
    n = tf.norm(x_f, axis=0)
    x_n = x_f / (tf.sqrt(tf.cast(out_dims, tf.float32)) * tf.reshape(n, (1, out_dims)))

    return tf.reshape(x_n, x.shape)

def unit_constraint(x):
    """ Return a constraint that clip `x` to the range [0, 1] """
    return tf.clip_by_value(x, 0.0, 1.0)

def conv2d(x, weights):
    """ Shortcut for `tf.nn.conv2d` """
    return tf.nn.conv2d(x, tf.cast(weights, tf.float16), (1, 1, 1, 1), 'SAME', True, 'NHWC')

def compute_gradients(loss, **kwargs):
    """ Shortcut to _tf.gradients_ (or similar alternatives) """

    var_list = tf.trainable_variables()
    grads = tf.gradients(
        loss,
        var_list,
        gate_gradients=True,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        #colocate_gradients_with_ops=True  # would force _custom gradients_ to the CPU
        **kwargs
    )

    return grads, var_list

def batch_norm(x, weights, mode, params, is_recomputing=False):
    """ Batch normalization layer. """
    num_channels = weights.shape[3]
    ones_op = tf.ones_initializer()
    zeros_op = tf.zeros_initializer()

    with tf.variable_scope(weights.op.name.split('/')[-1]):
        scale = tf.get_variable('scale', (num_channels,), tf.float32, ones_op, trainable=False, use_resource=True)
        mean = tf.get_variable('mean', (num_channels,), tf.float32, zeros_op, trainable=False, use_resource=True)
        variance = tf.get_variable('variance', (num_channels,), tf.float32, ones_op, trainable=False, use_resource=True)
        offset = tf.get_variable('offset', (num_channels,), tf.float32, zeros_op, trainable=True, use_resource=True)

    if not is_recomputing:
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
            weights,
            tf.reshape(scale / std_, (1, 1, 1, num_channels))
        )

        # fix the weights so that they appear in the _correct_ order according
        # to cuDNN (for NHWC):
        #
        # tensorflow: [h, w, in, out]
        # cudnn:      [out, h, w, in]
        weights_ = tf.transpose(weights_, [3, 0, 1, 2])

        tf.add_to_collection(DUMP_OPS, [offset, offset_, 'f2'])
        tf.add_to_collection(DUMP_OPS, [weights, weights_, 'f2'])

    def _forward(x):
        """ Returns the result of the forward inference pass on `x` """
        if mode == tf.estimator.ModeKeys.TRAIN:
            y, b_mean, b_variance = tf.nn.fused_batch_norm(
                x,
                scale,
                offset,
                None,
                None,
                data_format='NHWC',
                is_training=True
            )

            if not is_recomputing:
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
                data_format='NHWC',
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
    """
    init_op = orthogonal_initializer()
    half_op = tf.constant_initializer(0.5, tf.float32)
    num_channels = params['num_channels']

    conv_1 = tf.get_variable('conv_1', (3, 3, num_channels, num_channels), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)
    conv_2 = tf.get_variable('conv_2', (3, 3, num_channels, num_channels), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)
    alpha = tf.get_variable('alpha', (), tf.float32, half_op, constraint=unit_constraint, trainable=True, use_resource=True)

    tf.add_to_collection(DUMP_OPS, [alpha, alpha, 'f4'])

    def _forward(x, is_recomputing=False):
        """ Returns the result of the forward inference pass on `x` """

        # the 1st convolution
        y = batch_norm(conv2d(x, conv_1), conv_1, mode, params, is_recomputing=is_recomputing)
        y = tf.nn.relu(y)

        # the 2nd convolution
        y = batch_norm(conv2d(y, conv_2), conv_2, mode, params, is_recomputing=is_recomputing)
        y = tf.nn.relu(tf.cast(alpha, tf.float16) * y + tf.cast(1.0 - alpha, tf.float16) * x)

        return y

    return recompute_grad(_forward)(x)

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

    conv_1 = tf.get_variable('conv_1', (1, 1, num_channels, 1), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)
    linear_1 = tf.get_variable('linear_1', (361, 256), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)
    linear_2 = tf.get_variable('linear_2', (256, 1), tf.float32, init_op, use_resource=True)
    offset_1 = tf.get_variable('linear_1/offset', (256,), tf.float32, zeros_op, use_resource=True)
    offset_2 = tf.get_variable('linear_2/offset', (1,), tf.float32, zeros_op, use_resource=True)

    tf.add_to_collection(DUMP_OPS, [linear_1, linear_1, 'f2'])
    tf.add_to_collection(DUMP_OPS, [linear_2, linear_2, 'f2'])
    tf.add_to_collection(DUMP_OPS, [offset_1, offset_1, 'f2'])
    tf.add_to_collection(DUMP_OPS, [offset_2, offset_2, 'f2'])

    def _forward(x):
        """ Returns the result of the forward inference pass on `x` """
        y = batch_norm(conv2d(x, conv_1), conv_1, mode, params)
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

    conv_1 = tf.get_variable('conv_1', (1, 1, num_channels, 2), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)
    linear_1 = tf.get_variable('linear_1', (722, 362), tf.float32, init_op, use_resource=True)
    offset_1 = tf.get_variable('linear_1/offset', (362,), tf.float32, zeros_op, use_resource=True)

    tf.add_to_collection(DUMP_OPS, [linear_1, linear_1, 'f2'])
    tf.add_to_collection(DUMP_OPS, [offset_1, offset_1, 'f2'])

    def _forward(x):
        """ Returns the result of the forward inference pass on `x` """
        y = batch_norm(conv2d(x, conv_1), conv_1, mode, params)
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

    # store the number of channels in the JSON output so that we do not have to derive
    # this from the shape later.
    num_channels_ = tf.constant(num_channels, name='num_channels', dtype=tf.int32)

    tf.add_to_collection(DUMP_OPS, [num_channels_, num_channels_, 'i4'])

    with tf.variable_scope('01_upsample', reuse=tf.AUTO_REUSE):
        conv_1 = tf.get_variable('conv_1', (3, 3, num_inputs, num_channels), tf.float32, init_op, constraint=normalize_constraint, use_resource=True)

        y = batch_norm(conv2d(x, conv_1), conv_1, mode, params)
        y = tf.nn.relu(y)

    for i in range(num_blocks):
        with tf.variable_scope('{:02d}_residual'.format(2 + i), reuse=tf.AUTO_REUSE):
            y = residual_block(y, mode, params)

    # policy head
    with tf.variable_scope('{:02d}p_policy'.format(2 + num_blocks), reuse=tf.AUTO_REUSE):
        p = policy_head(y, mode, params)

    # value head
    with tf.variable_scope('{:02d}v_value'.format(2 + num_blocks), reuse=tf.AUTO_REUSE):
        v = value_head(y, mode, params)

    return v, p, y

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

    import os

    def __parse(line):
        result, example = get_single_example(line)

        if result != 0:
            features = np.zeros((19, 19, NUM_FEATURES), 'f2')
            boost = np.zeros((), 'f4')
            value = np.zeros((), 'f4')
            policy = np.zeros((362,), 'f2')
        else:
            features = np.frombuffer(example['features'], 'f4').astype('f2')
            value = np.asarray(1.0 if example['color'] == example['winner'] else -1.0, 'f4')
            policy = np.fromstring(base64.b85decode(example['policy']), 'f2')

            if example['number'] <= len(BOOST_PER_MOVE_NUMBER):
                boost = np.asarray(BOOST_PER_MOVE_NUMBER[example['number'] - 1], 'f4')
            else:
                boost = np.asarray(1.0, 'f4')

            # fix any partial policy
            policy[example['index']] += 1.0 - np.sum(policy)

        return features, boost, value, policy

    def _parse(line):
        return tuple(tf.py_func(
            __parse,
            [line],
            [tf.float16, tf.float32, tf.float32, tf.float16]
        ))

    def _illegal_policy(features, boost, value, policy):
        return tf.not_equal(tf.reduce_sum(boost), 0.0)

    def _fix_shape(features, boost, value, policy):
        features = tf.reshape(features, [19, 19, NUM_FEATURES])
        boost = tf.reshape(boost, [1])
        value = tf.reshape(value, [1])
        policy = tf.reshape(policy, [362])

        return features, boost, value, policy

    def _augment(features, boost, value, policy):
        def _identity(image):
            return tf.identity(image)

        def _flip_lr(image):
            return tf.reverse_v2(image, [1])

        def _flip_ud(image):
            return tf.reverse_v2(image, [0])

        def _transpose_main(image):
            return tf.transpose(image, [1, 0, 2])

        def _transpose_anti(image):
            return tf.reverse_v2(tf.transpose(image, [1, 0, 2]), [0, 1])

        def _rot90(image):
            return tf.transpose(tf.reverse_v2(image, [1]), [1, 0, 2])

        def _rot180(image):
            return tf.reverse_v2(image, [0, 1])

        def _rot270(image):
            return tf.reverse_v2(tf.transpose(image, [1, 0, 2]), [1])

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
        features = _apply_random(random, features)

        # transforming the policy is _harder_ since it has that extra pass
        # element at the end, so we temporarily remove it while the tensor gets
        # a random transformation applied
        policy, policy_pass = tf.split(policy, (361, 1))
        policy = tf.reshape(_apply_random(random, tf.reshape(policy, [19, 19, 1])), [361])
        policy = tf.concat([policy, policy_pass], 0)

        value = tf.reshape(value, [1])

        return features, boost, value, policy

    def _fix_history(features, boost, value, policy):
        """ Zeros out the history planes for 25% of the features. """
        zero_history_mask = np.asarray([1.0] * NUM_FEATURES, 'f2')
        zero_history_mask[3:4] = 0.0
        zero_history_mask = tf.constant(zero_history_mask, tf.float16, (1, 1, NUM_FEATURES))

        random = tf.random_uniform((), 0, 100, tf.int32)
        features = tf.case(
            [
                (tf.less(random, 10), lambda: features * zero_history_mask)
            ],
            default=lambda: features
        )

        return features, boost, value, policy

    with tf.device('cpu:0'):
        num_parallel_calls = max(os.cpu_count() - 8, 4)

        if len(files) > 1:
            file_names = tf.data.Dataset.from_tensor_slices(files)
            dataset = file_names.interleave(
                lambda file: tf.data.TextLineDataset(file),
                cycle_length=16,
                block_length=1,
                num_parallel_calls=16
            )
        else:
            dataset = tf.data.TextLineDataset(files)
        if is_training:
            dataset = dataset.shuffle(524288, reshuffle_each_iteration=True)
        dataset = dataset.map(_parse, num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(_illegal_policy)
        dataset = dataset.map(_fix_shape)
        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.map(_augment, num_parallel_calls=4)
            dataset = dataset.map(_fix_history, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(4)

        return dataset

def input_fn(files, batch_size, is_training):
    return get_dataset(files, batch_size, is_training).map(
        lambda features, boost, value, policy: (features, {'boost': boost, 'value': value, 'policy': policy})
    )

# -------- Model function --------

def model_fn(features, labels, mode, params):
    value_hat, policy_hat, tower_hat = tower(features, mode, params)

    if labels:
        # determine the loss for each of the components:
        #
        # - Value head
        # - Policy head
        #
        loss_value = tf.reshape(tf.squared_difference(
            tf.stop_gradient(labels['value']),
            value_hat
        ), (-1, 1))
        loss_policy = tf.reshape(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(labels['policy']),
            logits=policy_hat
        ), (-1, 1))

        loss_unboosted = loss_policy + 2.0 * loss_value
        loss = tf.reduce_mean(tf.stop_gradient(labels['boost']) * loss_unboosted)
        tf.add_to_collection(LOSS, loss_unboosted)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # set an initial learning rate and then rely on the `LearningRateScheduler`
            # hook to decrease it when the loss plateaus
            learning_rate = tf.Variable(params['learning_rate'], False, name='lr')
            loss_scale = 128

            tf.add_to_collection(LEARNING_RATE, learning_rate)

            global_step = tf.train.get_global_step()
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                gradients, variables = compute_gradients(loss_scale * loss)

                gradients = [grad / loss_scale for grad in gradients]
                train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)

            # during training it is very useful to plot the norm of the gradients at
            # each tensor so that we can detect the cause of any exploding gradients
            # or similar issues.
            for grad, var in zip(gradients, variables):
                var_name = var.op.name

                if grad is not None:
                    tf.summary.scalar('gradients/' + var_name, tf.norm(grad))
                tf.summary.scalar('norms/' + var_name, tf.norm(var))
                tf.summary.scalar('orthogonality/' + var_name, orthogonal_loss(var))

            tf.summary.scalar('learning_rate', learning_rate)
        else:
            train_op = None

        tf.summary.scalar('loss/policy', tf.reduce_mean(loss_policy))
        tf.summary.scalar('loss/value', tf.reduce_mean(loss_value))

        # evaluation metrics such as the accuracy is more human readable than
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

    # output the predictions, and some other intermediate tensors that may
    # be useful.
    predictions = {
        'features': features,
        'value': value_hat,
        'policy': tf.nn.softmax(policy_hat),
        'tower': tower_hat
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
    op_group.add_argument('--tower', action='store_true', help='print the final tower features to standard output')
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
    'learning_rate': 1e-4 if args.warm_start else 3e-4,

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
    steps_to_skip = 10000
    warm_start_from = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=args.warm_start[0],
        vars_to_warm_start='[0-9x].*'  # only layers
    )
else:
    steps_to_skip = 0
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
        hooks=hooks + [LearningRateScheduler(steps_to_skip)],
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
elif args.tower > 0:
    predictor = nn.predict(
        input_fn=lambda: input_fn(args.files, 1, False)
    )
    count = 0

    print('(;GM[1]FF[4]SZ[19]')
    for results in predictor:
        board_state = to_sgf_heatmap(results['features'], results['tower'])

        print('(;{})'.format(board_state))

        count += 1
        if count > 100:
            break
    print(')')
elif args.print:
    # tensors are given then print all available tensors with some statistics.
    if not args.files:
        out = {}

        for var in nn.get_variable_names():
            var_value = np.asarray(nn.get_variable_value(var))

            out[var] = {
                'mean': float(np.average(var_value)),
                'std': float(np.std(var_value)),
                'orthogonal': float(orthogonal_loss(var_value))
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
            print(var, nn.get_variable_value(var).tolist())

