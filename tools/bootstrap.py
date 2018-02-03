#!/usr/bin/env python3
# Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0103, C0301

"""
Train the network weights to better predict the played moves and winner
in the given dataset.

Usage: ./bootstrap.py <dataset...>
"""

from datetime import datetime
import math
import os
import sys

import tensorflow as tf
import numpy as np

class BatchNorm:
    """ Batch normalization layer. """

    def __init__(self, num_features, suffix=None, collection=None):
        if suffix is None:
            suffix = ''

        ones_op = tf.ones_initializer()
        zeros_op = tf.zeros_initializer()

        self._scale = tf.get_variable('scale'+suffix, (num_features,), tf.float32, ones_op, trainable=False)
        self._offset = tf.get_variable('offset'+suffix, (num_features,), tf.float32, zeros_op, trainable=True)
        self._mean = tf.get_variable('mean'+suffix, (num_features,), tf.float32, zeros_op, trainable=False)
        self._variance = tf.get_variable('variance'+suffix, (num_features,), tf.float32, ones_op, trainable=False)

        #tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._scale)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._offset)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._mean)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._variance)
        tf.add_to_collection(tf.GraphKeys.BIASES, self._offset)

        if collection:
            tf.add_to_collection(collection, self._offset)
            tf.add_to_collection(collection, self._mean)
            tf.add_to_collection(collection, self._variance)

    def dump(self, sess, weights, into=None):
        """
        Returns a dictionary that contains this batch normalization layer folded
        into the given convolution weights.

        The convolution weights will also get transposed to the format expected
        by cuDNN.
        """

        if into is None:
            into = {}

        # fix the weights so that they appear in the _correct_ order according
        # to cuDNN.
        #
        # tensorflow: [h, w, in, out]
        # cudnn:      [out, in, h, w]
        weights_ = np.transpose(sess.run(weights), [3, 2, 0, 1])

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
        [mean, variance, offset] = sess.run([self._mean, self._variance, self._offset])

        assert weights_.shape[0] == mean.shape[0]
        assert variance.shape[0] == mean.shape[0]

        into[self._offset] = offset - mean / np.sqrt(variance + 0.001)
        into[weights] = np.multiply(
            weights_,
            np.reshape(1.0 / np.sqrt(variance + 0.001), (weights_.shape[0], 1, 1, 1))
        )

        return into

    def __call__(self, x, is_training=True):
        if is_training:
            y, b_mean, b_variance = tf.nn.fused_batch_norm(
                x,
                self._scale,
                self._offset,
                None,
                None,
                data_format='NCHW',
                is_training=True
            )

            with tf.device(None):
                update_mean_op = tf.assign_sub(self._mean, 0.01 * (self._mean - b_mean), use_locking=True)
                update_variance_op = tf.assign_sub(self._variance, 0.01 * (self._variance - b_variance), use_locking=True)

                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean_op)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance_op)
        else:
            y, _, _ = tf.nn.fused_batch_norm(
                x,
                self._scale,
                self._offset,
                self._mean,
                self._variance,
                data_format='NCHW',
                is_training=False
            )

        return y

class ResidualBlock:
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

    def __init__(self, num_features, collection=None):
        glorot_op = tf.glorot_normal_initializer()

        self._conv_1 = tf.get_variable('weights_1', (3, 3, num_features, num_features), tf.float32, glorot_op)
        self._bn_1 = BatchNorm(num_features, '_1')
        self._conv_2 = tf.get_variable('weights_2', (3, 3, num_features, num_features), tf.float32, glorot_op)
        self._bn_2 = BatchNorm(num_features, '_2')

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._conv_1)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._conv_2)

        if collection:
            tf.add_to_collection(collection, self._conv_1)
            tf.add_to_collection(collection, self._conv_2)

    def dump(self, sess, into=None):
        """ Returns a dictionary that contains all model variables of this block. """

        if into is None:
            into = {}

        self._bn_1.dump(sess, self._conv_1, into=into)
        self._bn_2.dump(sess, self._conv_2, into=into)

        return into

    def __call__(self, x, is_training=True):
        y = tf.nn.conv2d(x, self._conv_1, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn_1(y, is_training)
        y = tf.nn.relu(y)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tf.identity(y, 'output_1'))

        y = tf.nn.conv2d(y, self._conv_2, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn_2(y, is_training)
        y = tf.nn.relu(y + x)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tf.identity(y, 'output_2'))

        return y

class ValueHead:
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

    VARIABLES = 'value_variables'

    def __init__(self, num_features):
        glorot_op = tf.glorot_normal_initializer()
        zeros_op = tf.zeros_initializer()

        self._downsample = tf.get_variable('downsample', (1, 1, num_features, 1), tf.float32, glorot_op)
        self._bn = BatchNorm(1, collection=ValueHead.VARIABLES)
        self._weights_1 = tf.get_variable('weights_1', (361, 256), tf.float32, glorot_op)
        self._weights_2 = tf.get_variable('weights_2', (256, 1), tf.float32, glorot_op)
        self._bias_1 = tf.get_variable('bias_1', (256,), tf.float32, zeros_op)
        self._bias_2 = tf.get_variable('bias_2', (1,), tf.float32, zeros_op)

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._downsample)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._weights_1)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._weights_2)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._bias_1)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._bias_2)

        tf.add_to_collection(ValueHead.VARIABLES, self._downsample)
        tf.add_to_collection(ValueHead.VARIABLES, self._weights_1)
        tf.add_to_collection(ValueHead.VARIABLES, self._weights_2)
        tf.add_to_collection(ValueHead.VARIABLES, self._bias_1)
        tf.add_to_collection(ValueHead.VARIABLES, self._bias_2)

        tf.add_to_collection(tf.GraphKeys.BIASES, self._bias_1)
        tf.add_to_collection(tf.GraphKeys.BIASES, self._bias_2)

    def dump(self, sess, into=None):
        """ Returns a dictionary that contains all model variables of this head. """

        if into is None:
            into = {}

        self._bn.dump(sess, self._downsample, into=into)

        into[self._weights_1] = sess.run(self._weights_1)
        into[self._weights_2] = sess.run(self._weights_2)
        into[self._bias_1] = sess.run(self._bias_1)
        into[self._bias_2] = sess.run(self._bias_2)

        return into

    def __call__(self, x, is_training=True):
        y = tf.nn.conv2d(x, self._downsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn(y, is_training)
        y = tf.nn.relu(y)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tf.identity(y, 'output_1'))

        y = tf.reshape(y, (-1, 361))
        y = tf.matmul(y, self._weights_1) + self._bias_1
        y = tf.nn.relu(y)
        y = tf.matmul(y, self._weights_2) + self._bias_2

        return tf.nn.tanh(y)

class PolicyHead:
    """
    The policy head attached after the residual blocks as described by DeepMind:

    1. A convolution of 2 filters of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer that outputs a vector of size 19**2 + 1 = 362 corresponding to
       logit probabilities for all intersections and the pass move
    """

    VARIABLES = 'policy_variables'

    def __init__(self, num_features):
        glorot_op = tf.glorot_normal_initializer()
        zeros_op = tf.zeros_initializer()

        self._downsample = tf.get_variable('downsample', (1, 1, num_features, 2), tf.float32, glorot_op)
        self._bn = BatchNorm(2, collection=PolicyHead.VARIABLES)
        self._weights = tf.get_variable('weights', (722, 362), tf.float32, glorot_op)
        self._bias = tf.get_variable('bias', (362,), tf.float32, zeros_op)

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._downsample)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._weights)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._bias)

        tf.add_to_collection(PolicyHead.VARIABLES, self._downsample)
        tf.add_to_collection(PolicyHead.VARIABLES, self._weights)
        tf.add_to_collection(PolicyHead.VARIABLES, self._bias)

        tf.add_to_collection(tf.GraphKeys.BIASES, self._bias)

    def dump(self, sess, into=None):
        """ Returns a dictionary that contains all model variables of this head. """

        if into is None:
            into = {}

        self._bn.dump(sess, self._downsample, into=into)

        into[self._weights] = sess.run(self._weights)
        into[self._bias] = sess.run(self._bias)

        return into

    def __call__(self, x, is_training=True):
        y = tf.nn.conv2d(x, self._downsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn(y, is_training)
        y = tf.nn.relu(y)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tf.identity(y, 'output_1'))

        y = tf.reshape(y, (-1, 722))
        y = tf.matmul(y, self._weights) + self._bias

        return y

class Tower:
    """
    The full neural network used to predict the value and policy tensors for a mini-batch of board
    positions.
    """

    VARIABLES = 'tower_variables'

    def __init__(self, num_features=128):
        glorot_op = tf.glorot_normal_initializer()

        with tf.variable_scope('01_upsample') as self._upsample_scope:
            self._upsample = tf.get_variable('weights', (3, 3, 36, num_features), tf.float32, glorot_op)
            self._bn = BatchNorm(num_features, collection=Tower.VARIABLES)

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._upsample)
        tf.add_to_collection(Tower.VARIABLES, self._upsample)

        # residual blocks
        self._residual_scopes = [None] * 9
        self._residuals = []

        for i in range(9):
            with tf.variable_scope('{:02d}_residual'.format(2 + i)) as self._residual_scopes[i]:
                self._residuals += [ResidualBlock(num_features, collection=Tower.VARIABLES)]

        # policy head
        with tf.variable_scope('11p_policy') as self._policy_scope:
            self._policy = PolicyHead(num_features)

        # value head
        with tf.variable_scope('11v_value') as self._value_scope:
            self._value = ValueHead(num_features)

    def dump(self, sess, into=None):
        """ Returns a dictionary that contains all model variables of this tower. """

        if into is None:
            into = {}

        self._bn.dump(sess, self._upsample, into=into)

        for residual in self._residuals:
            residual.dump(sess, into=into)
        self._policy.dump(sess, into=into)
        self._value.dump(sess, into=into)

        return into

    def __call__(self, x, is_training=True, train_tower=True, train_policy=True, train_value=True):
        with tf.name_scope(self._upsample_scope.original_name_scope):
            y = tf.nn.conv2d(x, self._upsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
            y = self._bn(y, is_training and train_tower)
            y = tf.nn.relu(y)

            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tf.identity(y, 'output'))

        for (i, resb) in enumerate(self._residuals):
            with tf.name_scope(self._residual_scopes[i].original_name_scope):
                y = resb(y, is_training and train_tower)

        with tf.name_scope(self._policy_scope.original_name_scope):
            p = self._policy(y, is_training and train_policy)

        with tf.name_scope(self._value_scope.original_name_scope):
            v = self._value(y, is_training and train_value)

        return v, p


def list_local_devices():
    """ Returns a generator of all local GPU devices on this machine. """
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()

    return [dev.name for dev in local_device_protos if dev.device_type == 'GPU']


def cosine_decay_restarts(learning_rate, global_step, first_decay_steps, alpha=0.0):
    """
    Applies cosine decay with restarts to the learning rate.

    See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983
    """
    with tf.name_scope(None, "SGDRDecay", [learning_rate, global_step]):
        global_step = tf.cast(global_step, tf.float32)
        first_decay_steps = tf.cast(first_decay_steps, tf.float32)
        alpha = tf.cast(alpha, tf.float32)
        t_mul = 1.4  # used to derive the number of iterations in the i-th period
        m_mul = 0.5  # used to derive the initial learning rate of the i-th period:

        completed_fraction = global_step / first_decay_steps
        i_restart = tf.floor(tf.log(1.0 - completed_fraction * (1.0 - t_mul)) / tf.log(t_mul))

        sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
        completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart

        m_fac = m_mul ** i_restart
        cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(math.pi * completed_fraction))
        decayed = (1 - alpha) * cosine_decayed + alpha

        return learning_rate * decayed


def make_dataset_iterator(files, batch_size=1):
    """ Returns a tf.DataSet initializable iterator over the given files """

    dataset = tf.data.FixedLengthRecordDataset(files, 26718)
    dataset = dataset.map(lambda x: tf.cast(tf.decode_raw(x, tf.half), tf.float32))
    dataset = dataset.map(lambda x: tf.split(x, (12996, 1, 362)))
    dataset = dataset.shuffle(196704)
    dataset = dataset.batch(batch_size if 'BATCH_SIZE' not in os.environ else int(os.environ['BATCH_SIZE']))

    return dataset.make_initializable_iterator()


def main(files, reset=False, reset_lr=False, only_tower=False, only_policy=False, only_value=False):
    """ Main function """

    iterator = make_dataset_iterator(files, batch_size=512)

    with tf.device('cpu:0'):
        global_step = tf.train.create_global_step()
        epoch = tf.get_variable('epoch', (), tf.int64, tf.zeros_initializer(), trainable=False)
        epoch_op = tf.assign_add(epoch, 1)

    # setup the forward pass while keeping track of what variables to train, which
    # becomes more annoying because of batch normalization
    train_all = not only_tower and not only_policy and not only_value
    _tower = Tower()

    original_trainable = set(tf.trainable_variables())

    if not train_all:
        tf.get_default_graph().clear_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        if only_tower:
            for var in tf.get_collection(Tower.VARIABLES):
                if var in original_trainable:
                    tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var)
        if only_policy:
            for var in tf.get_collection(PolicyHead.VARIABLES):
                if var in original_trainable:
                    tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var)
        if only_value:
            for var in tf.get_collection(ValueHead.VARIABLES):
                if var in original_trainable:
                    tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var)

    # distribute the work over all of the local GPU's
    value_losses, policy_losses = [], []
    policy_accuracy_1s, policy_accuracy_3s, policy_accuracy_5s, value_accuracies = [], [], [], []

    for (i, dev) in enumerate(list_local_devices()):
        var_scope = tf.get_variable_scope()
        name_scope = 'gpu_' + str(i)

        with tf.variable_scope(var_scope, reuse=True), tf.device(dev), tf.name_scope(name_scope):
            # re-use the variable that were created in the beginning instead of re-allocating
            # them for each tower
            tower = Tower()

            # create a local model and the put it away
            with tf.device(None):
                features, value, policy = iterator.get_next()

            features = tf.reshape(features, (-1, 36, 19, 19))
            value_hat, policy_hat = tower(
                features,
                train_tower=train_all or only_tower,
                train_policy=train_all or only_policy,
                train_value=train_all or only_value
            )

            with tf.device(None):
                if i == 0:
                    # log the result of the forward pass only from the first GPU since
                    # we are not interested in exact results anyway
                    #tf.summary.histogram('value', value)
                    #tf.summary.histogram('value_hat', value_hat)
                    #tf.summary.histogram('policy', policy)
                    #tf.summary.histogram('policy_hat', policy_hat)
                    pass

                policy_argmax = tf.argmax(policy, axis=1)
                policy_accuracy_1s.append(tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_argmax, k=1), tf.float32)))
                policy_accuracy_3s.append(tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_argmax, k=3), tf.float32)))
                policy_accuracy_5s.append(tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_argmax, k=5), tf.float32)))
                value_accuracies.append(tf.reduce_mean(tf.cast(tf.equal(tf.sign(value), tf.sign(value_hat)), tf.float32)))

            # setup the loss for the local model
            policy_losses.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(policy),
                logits=policy_hat
            )))
            value_losses.append(tf.reduce_mean(tf.squared_difference(value, value_hat)))

    # gather the losses and variables from all of the tower into a single loss function
    # that gets forwarded to the optimizer
    policy_loss = tf.reduce_mean(policy_losses)
    value_loss = tf.reduce_mean(value_losses)
    reg_loss = tf.reduce_sum([
        tf.nn.l2_loss(var)
        for var in original_trainable
        if var not in tf.get_collection(tf.GraphKeys.BIASES)  # do not apply to bias terms
    ])

    loss = policy_loss + 0.1 * value_loss + 4e-4 * reg_loss

    tf.summary.scalar('loss/policy', policy_loss)
    tf.summary.scalar('loss/value', value_loss)
    tf.summary.scalar('loss/regularization', reg_loss)
    tf.summary.scalar('loss', loss)

    learning_rate = cosine_decay_restarts(0.1, global_step, 1500)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    optimizer_op = optimizer.minimize(
        loss,
        global_step=global_step,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        colocate_gradients_with_ops=True
    )

    # summaries for debugging purposes
    tf.summary.scalar('accuracy/policy_1', tf.reduce_mean(policy_accuracy_1s))
    tf.summary.scalar('accuracy/policy_3', tf.reduce_mean(policy_accuracy_3s))
    tf.summary.scalar('accuracy/policy_5', tf.reduce_mean(policy_accuracy_5s))
    tf.summary.scalar('accuracy/value', tf.reduce_mean(value_accuracies))

    #for var in tf.model_variables():
    #    tf.summary.histogram(var.name, var)

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('epoch', epoch)

    # operations
    summary_writer = tf.summary.FileWriter('logs/' + datetime.now().strftime('%Y%m%d.%H%M') + '/', graph=tf.get_default_graph())
    summary_op = tf.summary.merge_all()
    update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    saver_vars = list(set(tf.model_variables() + [global_step, epoch]))
    saver = tf.train.Saver(saver_vars, keep_checkpoint_every_n_hours=2)

    #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

        # restore model from checkpoint
        latest_checkpoint = tf.train.latest_checkpoint('models/')
        if latest_checkpoint is not None:
            print('Restoring from ' + latest_checkpoint)

            saver.restore(sess, latest_checkpoint)

        # reset the relevant parts of the graph. This is mostly useful when re-training only
        # parts of the graph
        if reset:
            if only_tower:
                print('Reset intermediate tower variables')
                sess.run([tf.variables_initializer(tf.get_collection(Tower.VARIABLES))])
            if only_policy:
                print('Reset policy head variables')
                sess.run([tf.variables_initializer(tf.get_collection(PolicyHead.VARIABLES))])
            if only_value:
                print('Reset value head variables')
                sess.run([tf.variables_initializer(tf.get_collection(ValueHead.VARIABLES))])
        if reset_lr:
            print('Reset the learning rate')
            sess.run([tf.assign(global_step, 0, use_locking=True)])

        sess.graph.finalize()

        while True:
            sess.run(iterator.initializer)

            try:
                while True:
                    global_step_hat, _, _ = sess.run([global_step, optimizer_op, update_op])

                    if global_step_hat % 25 == 0:
                        summary_hat = sess.run(summary_op)
                        summary_writer.add_summary(summary_hat, global_step_hat)
                    if global_step_hat > 0 and global_step_hat % 1000 == 0:
                        saver.save(sess, 'models/dream-go', global_step=global_step_hat, write_meta_graph=False)
            except KeyboardInterrupt:
                break  # quit
            except:
                epoch = sess.run(epoch_op)
                if epoch >= 14:
                    break

                saver.save(sess, 'models/dream-go', global_step=global_step_hat, write_meta_graph=False)

        # save the model
        saver.save(sess, 'models/dream-go', global_step=global_step_hat, write_meta_graph=False)


def verify(args):
    """ Retrieve accuracy for a verification test-set. """

    iterator = make_dataset_iterator(args, batch_size=1)

    # get the answer from the data-set and the prediction
    tower = Tower()
    features, value, policy = iterator.get_next()
    features = tf.reshape(features, (-1, 36, 19, 19))
    value_hat, policy_hat = tower(features, is_training=False)

    policy_argmax = tf.argmax(policy, axis=1)
    policy_accuracy_1 = tf.cast(tf.nn.in_top_k(policy_hat, policy_argmax, k=1), tf.float32)
    policy_accuracy_3 = tf.cast(tf.nn.in_top_k(policy_hat, policy_argmax, k=3), tf.float32)
    policy_accuracy_5 = tf.cast(tf.nn.in_top_k(policy_hat, policy_argmax, k=5), tf.float32)
    value_accuracy = tf.cast(tf.equal(tf.sign(value), tf.sign(value_hat)), tf.float32)

    # restore only model variables
    saver_vars = tf.model_variables()
    saver = tf.train.Saver(saver_vars, keep_checkpoint_every_n_hours=2)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        #from tensorflow.python import debug as tf_debug
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
        sess.graph.finalize()

        # restore model from checkpoint
        latest_checkpoint = tf.train.latest_checkpoint('models/')
        if latest_checkpoint is not None:
            print('Restoring from ' + latest_checkpoint)

            saver.restore(sess, latest_checkpoint)

        # loop over the entire data-set
        sess.run(iterator.initializer)

        accuracy_p1 = 0
        accuracy_p3 = 0
        accuracy_p5 = 0
        accuracy_v = 0
        count = 0

        while True:
            try:
                [policy_1, policy_3, policy_5, value_1] = sess.run([
                    policy_accuracy_1,
                    policy_accuracy_3,
                    policy_accuracy_5,
                    value_accuracy
                ])

                count += 1
                accuracy_p1 = accuracy_p1 + (policy_1 - accuracy_p1) / count
                accuracy_p3 = accuracy_p3 + (policy_3 - accuracy_p3) / count
                accuracy_p5 = accuracy_p5 + (policy_5 - accuracy_p5) / count
                accuracy_v = accuracy_v + (value_1 - accuracy_v) / count
            except (tf.errors.OutOfRangeError, KeyboardInterrupt):
                break

        print('policy (top 1/3/5): {:.1f}%/{:.1f}%/{:.1f}%'.format(
            100.0 * np.asscalar(accuracy_p1),
            100.0 * np.asscalar(accuracy_p3),
            100.0 * np.asscalar(accuracy_p5)
        ))
        print('value: {:.1f}%'.format(
            100.0 * np.asscalar(accuracy_v)
        ))


def calibrate(sess, tower, files):
    """
    Calculate the optimal scaling of each activation using cross-entropy
    cost. This scale is used for int8 weights and activations.
    """

    # setup the iterator over all features in the gives files
    iterator = make_dataset_iterator(files, batch_size=32)
    features, _value, _policy = iterator.get_next()
    features = tf.reshape(features, (-1, 36, 19, 19))
    _value_hat, _policy_hat = tower(features, is_training=False)

    # pre-allocate the dictionaries containing all activations
    histogram = {}
    min_boundary = {}
    max_boundary = {}

    try:
        import base64
        from scipy import stats

        # loop over the entire dataset to collect [min, max] of each activation
        min_ops = {}
        max_ops = {}

        for act in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
            min_ops[act.name] = tf.reduce_min(act)
            max_ops[act.name] = tf.reduce_max(act)

        sess.run(iterator.initializer)

        while True:
            try:
                [min_hat, max_hat] = sess.run([min_ops, max_ops])

                for (name, v) in min_hat.items():
                    min_boundary[name] = min(min_boundary.get(name, math.inf), v)
                for (name, v) in max_hat.items():
                    max_boundary[name] = max(max_boundary.get(name, -math.inf), v + 1e-2)
            except tf.errors.OutOfRangeError:
                break

        # loop over the entire dataset to collect the histograms, where each
        # histogram is within the minimum and maximum we just collected
        histogram_ops = {}
        num_bins = 65536

        for act in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
            assert min_boundary[act.name] == 0.0

            histogram_ops[act.name] = tf.histogram_fixed_width(
                act,
                [0.0, max_boundary[act.name]],
                nbins=num_bins + 2,
                dtype=tf.int64
            )

        sess.run(iterator.initializer)

        while True:
            try:
                histogram_hat = sess.run(histogram_ops)

                for (name, h) in histogram_hat.items():
                    histogram[name] = histogram.get(name, 0) + h
            except tf.errors.OutOfRangeError:
                break

        # normalize the histograms (in double precision), we throw away
        # the first and the last elements so that the histogram cover
        # the correct range.
        for name in histogram:
            h = histogram[name][1:-1].astype('f8')  # cast to double

            histogram[name] = h / np.linalg.norm(h, 1)

        # find the optimal scale for each variable by using the
        # kullback–leibler divergence
        values = {}

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

        for name in histogram:
            best_i = min(
                range(128, num_bins + 128, 128),
                key=lambda i: _entropy_calibrate(histogram[name], i)
            )

            scale = (best_i + 0.5) * (max_boundary[name] / num_bins)
            scale = np.asarray(scale, 'f2').tostring()

            values[name] = {
                's': base64.b85encode(scale, pad=True).decode('ascii')
            }

            # print a dot as a status message
            print('.', end='', flush=True, file=sys.stderr)

        return values
    except KeyboardInterrupt:
        return {}


def dump(args):
    """
    Dump the given (or latest if none is given) checkpoint as a JSON file that
    is readable by dream-go
    """
    tower = Tower()

    # restore only model variables
    saver_vars = tf.model_variables()
    saver = tf.train.Saver(saver_vars, keep_checkpoint_every_n_hours=2)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

        checkpoints = [arg for arg in args if not arg.endswith('.bin')]

        if not checkpoints:
            latest_checkpoint = tf.train.latest_checkpoint('models/')
            if latest_checkpoint is not None:
                saver.restore(sess, latest_checkpoint)
        else:
            saver.restore(sess, checkpoints[0])

        # gather histograms over all of the activations for int8 calibration
        files = [arg for arg in args if arg.endswith('.bin')]
        values = {}

        if files:
            values = calibrate(sess, tower, files)

        # dump the variables to JSON in half precision in order to save
        # save disk space.
        import base64
        import json

        for (var, value) in tower.dump(sess).items():
            serialized = value.flatten().astype('f2').tostring()
            scale = np.linalg.norm(value.flatten(), math.inf).astype('f2').tostring()

            values[var.name] = {
                's': base64.b85encode(scale, pad=True).decode('ascii'),
                'v': base64.b85encode(serialized, pad=True).decode('ascii')
            }

        json.dump(values, sys.stdout, sort_keys=True)


if __name__ == '__main__':
    options = [arg for arg in sys.argv[1:] if arg.startswith('--')]
    rest = [arg for arg in sys.argv[1:] if not arg.startswith('--')]

    if '--dump' in options:
        dump(rest)
    elif not rest:
        print('Usage: bootstrap.py [options] <data...>')
        print()
        print('    --dump         Dump the current weights to STDOUT')
        print('    --verify       Dump the accuracy of the given data to STDOUT')
        print('    --reset        Reset the weights to their initial value')
        print('    --reset-lr     Reset the learning rate')
        print('    --only-tower   Only train or reset the intermediate tower')
        print('    --only-policy  Only train or reset the policy head')
        print('    --only-value   Only train or reset the value head')
        print()
        quit()
    elif '--verify' in options:
        verify(rest)
    else:
        main(
            rest,  # dataset
            reset=('--reset' in options),
            reset_lr=('--reset-lr' in options),
            only_tower=('--only-tower' in options),
            only_policy=('--only-policy' in options),
            only_value=('--only-value' in options)
        )
