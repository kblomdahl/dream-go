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
from math import nan, isnan
import sys
import os

import tensorflow as tf
import numpy as np

class BatchNorm:
    """ Batch normalization layer. """

    def __init__(self, num_features, suffix=None):
        if suffix is None:
            suffix = ''

        ones_op = tf.ones_initializer()
        zeros_op = tf.zeros_initializer()

        self._scale = tf.get_variable('scale'+suffix, (num_features,), tf.float32, ones_op, trainable=False)
        self._offset = tf.get_variable('offset'+suffix, (num_features,), tf.float32, zeros_op, trainable=False)
        self._mean = tf.get_variable('mean'+suffix, (num_features,), tf.float32, zeros_op, trainable=False)
        self._variance = tf.get_variable('variance'+suffix, (num_features,), tf.float32, ones_op, trainable=False)

        #tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._scale)
        #tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._offset)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._mean)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._variance)

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

            update_mean_op = tf.assign_sub(self._mean, 0.01 * (self._mean - b_mean))
            update_variance_op = tf.assign_sub(self._variance, 0.01 * (self._variance - b_variance))

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

    def __init__(self, num_features):
        glorot_op = tf.glorot_normal_initializer()

        self._conv_1 = tf.get_variable('weights_1', (3, 3, num_features, num_features), tf.float32, glorot_op)
        self._bn_1 = BatchNorm(num_features, '_1')
        self._conv_2 = tf.get_variable('weights_2', (3, 3, num_features, num_features), tf.float32, glorot_op)
        self._bn_2 = BatchNorm(num_features, '_2')

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._conv_1)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._conv_2)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, self._conv_1)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, self._conv_2)

    def __call__(self, x, is_training=True):
        y = tf.nn.conv2d(x, self._conv_1, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn_1(y, is_training)
        y = tf.nn.relu(y)
        y = tf.nn.conv2d(y, self._conv_2, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn_2(y, is_training)

        return tf.nn.relu(y + x)

class ValueHead:
    """
    The value head attached after the residual blocks as described by DeepMind:

    1. A convolution of 1 filter of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer to a hidden layer of size 256
    5. A rectifier non-linearity
    6. A fully connected linear layer to a scalar
    7. A tanh non-linearity outputting a scalar in the range [−1, 1]
    """

    def __init__(self, num_features):
        glorot_op = tf.glorot_normal_initializer()
        zeros_op = tf.zeros_initializer()

        self._downsample = tf.get_variable('downsample', (1, 1, num_features, 1), tf.float32, glorot_op)
        self._bn = BatchNorm(1)
        self._weights_1 = tf.get_variable('weights_1', (361, 256), tf.float32, glorot_op)
        self._weights_2 = tf.get_variable('weights_2', (256, 1), tf.float32, glorot_op)
        self._bias_1 = tf.get_variable('bias_1', (256,), tf.float32, zeros_op)
        self._bias_2 = tf.get_variable('bias_2', (1,), tf.float32, zeros_op)

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._downsample)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._weights_1)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._weights_2)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._bias_1)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._bias_2)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, self._downsample)

    def __call__(self, x, is_training=True):
        y = tf.nn.conv2d(x, self._downsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn(y, is_training)
        y = tf.nn.relu(y)
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

    def __init__(self, num_features):
        glorot_op = tf.glorot_normal_initializer()
        zeros_op = tf.zeros_initializer()

        self._downsample = tf.get_variable('downsample', (1, 1, num_features, 2), tf.float32, glorot_op)
        self._bn = BatchNorm(2)
        self._weights = tf.get_variable('weights', (722, 362), tf.float32, glorot_op)
        self._bias = tf.get_variable('bias', (362,), tf.float32, zeros_op)

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._downsample)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._weights)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._bias)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, self._downsample)

    def __call__(self, x, is_training=True):
        y = tf.nn.conv2d(x, self._downsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn(y, is_training)
        y = tf.nn.relu(y)
        y = tf.reshape(y, (-1, 722))
        y = tf.matmul(y, self._weights) + self._bias

        return y

class Tower:
    """
    The full neural network used to predict the value and policy tensors for a mini-batch of board
    positions.
    """

    def __init__(self, num_features=256):
        glorot_op = tf.glorot_normal_initializer()

        with tf.variable_scope('01_upsample'):
            self._upsample = tf.get_variable('weights', (3, 3, 34, num_features), tf.float32, glorot_op)
            self._bn = BatchNorm(num_features)

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._upsample)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, self._upsample)

        # residual blocks
        self._residuals = []

        for i in range(19):
            with tf.variable_scope('{:02d}_residual'.format(2 + i)):
                self._residuals += [ResidualBlock(num_features)]

        # policy head
        with tf.variable_scope('21p_policy'):
            self._policy = PolicyHead(num_features)

        # value head
        with tf.variable_scope('21v_value'):
            self._value = ValueHead(num_features)

    def __call__(self, x, is_training=True):
        y = tf.nn.conv2d(x, self._upsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn(y, is_training)
        y = tf.nn.relu(y)

        for resb in self._residuals:
            y = resb(y, is_training)

        p = self._policy(y, is_training)
        v = self._value(y, is_training)

        return v, p

def main(args):
    """ Main function """

    dataset = tf.data.FixedLengthRecordDataset(args, 25274)
    dataset = dataset.map(lambda x: tf.cast(tf.decode_raw(x, tf.half), tf.float32))
    dataset = dataset.map(lambda x: tf.split(x, (12274, 1, 362)))
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(256 if 'BATCH_SIZE' not in os.environ else int(os.environ['BATCH_SIZE']))
    iterator = dataset.make_initializable_iterator()

    #
    tower = Tower()

    with tf.device('cpu:0'):
        global_step = tf.train.create_global_step()
        epoch = tf.get_variable('epoch', (), tf.int64, tf.zeros_initializer(), trainable=False)
        epoch_op = tf.assign_add(epoch, 1)

    features, value, policy = iterator.get_next()
    features = tf.reshape(features, (-1, 34, 19, 19))
    value_hat, policy_hat = tower(features)

    #
    policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=policy_hat))
    value_loss = tf.reduce_mean(tf.squared_difference(value, value_hat))
    reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    loss = policy_loss + value_loss + 1e-4 * reg_loss

    tf.summary.scalar('loss/policy', policy_loss)
    tf.summary.scalar('loss/value', value_loss)
    tf.summary.scalar('loss/regularization', reg_loss)
    tf.summary.scalar('loss', loss)

    learning_rate = tf.train.piecewise_constant(
        global_step,
        [50000, 100000, 300000, 500000, 700000],
        [1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    )
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    optimizer_op = optimizer.minimize(loss, global_step=global_step)

    # summaries
    policy_accuracy_1 = tf.cast(tf.nn.in_top_k(policy_hat, tf.argmax(policy, axis=1), k=1), tf.float32)
    policy_accuracy_3 = tf.cast(tf.nn.in_top_k(policy_hat, tf.argmax(policy, axis=1), k=3), tf.float32)
    policy_accuracy_5 = tf.cast(tf.nn.in_top_k(policy_hat, tf.argmax(policy, axis=1), k=5), tf.float32)
    value_accuracy = tf.cast(tf.equal(tf.sign(value), tf.sign(value_hat)), tf.float32)

    tf.summary.scalar('accuracy/policy_1', tf.reduce_mean(policy_accuracy_1))
    tf.summary.scalar('accuracy/policy_3', tf.reduce_mean(policy_accuracy_3))
    tf.summary.scalar('accuracy/policy_5', tf.reduce_mean(policy_accuracy_5))
    tf.summary.scalar('accuracy/value', tf.reduce_mean(value_accuracy))

    for var in tf.model_variables():
        tf.summary.histogram(var.name, var)

    tf.summary.histogram('value', value)
    tf.summary.histogram('value_hat', value_hat)
    tf.summary.histogram('policy', policy)
    tf.summary.histogram('policy_hat', policy_hat)

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('epoch', epoch)

    # operations
    summary_writer = tf.summary.FileWriter('logs/' + datetime.now().strftime('%Y%m%d.%H%M') + '/', graph=tf.get_default_graph())
    summary_op = tf.summary.merge_all()
    update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    saver_vars = tf.model_variables() + [global_step, epoch]
    saver = tf.train.Saver(saver_vars, keep_checkpoint_every_n_hours=2)

    #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
        sess.graph.finalize()

        # restore model from checkpoint
        latest_checkpoint = tf.train.latest_checkpoint('models/')
        if latest_checkpoint is not None:
            print('Restoring from ' + latest_checkpoint)

            saver.restore(sess, latest_checkpoint)

        while True:
            sess.run(iterator.initializer)

            try:
                while True:
                    global_step_hat, _, _ = sess.run([global_step, optimizer_op, update_op])

                    if global_step_hat > 0 and global_step_hat % 10 == 0:
                        summary_hat = sess.run(summary_op)
                        summary_writer.add_summary(summary_hat, global_step_hat)
                    if global_step_hat > 0 and global_step_hat % 1000 == 0:
                        saver.save(sess, 'models/dream-go', global_step=global_step_hat, write_meta_graph=False)
            except KeyboardInterrupt:
                break  # quit
            except:
                sess.run([epoch_op])
                saver.save(sess, 'models/dream-go', global_step=global_step_hat, write_meta_graph=False)

        # save the model
        saver.save(sess, 'models/dream-go', global_step=global_step_hat, write_meta_graph=False)

def verify(args):
    """ Retrieve accuracy for a verification test-set. """

    dataset = tf.data.FixedLengthRecordDataset(args, 25274)
    dataset = dataset.map(lambda x: tf.cast(tf.decode_raw(x, tf.half), tf.float32))
    dataset = dataset.map(lambda x: tf.split(x, (12274, 1, 362)))
    dataset = dataset.batch(1)
    iterator = dataset.make_initializable_iterator()

    # get the answer from the data-set and the prediction
    tower = Tower()
    features, value, policy = iterator.get_next()
    features = tf.reshape(features, (-1, 34, 19, 19))
    value_hat, policy_hat = tower(features, is_training=False)

    policy_accuracy_1 = tf.cast(tf.nn.in_top_k(policy_hat, tf.argmax(policy, axis=1), k=1), tf.float32)
    policy_accuracy_3 = tf.cast(tf.nn.in_top_k(policy_hat, tf.argmax(policy, axis=1), k=3), tf.float32)
    policy_accuracy_5 = tf.cast(tf.nn.in_top_k(policy_hat, tf.argmax(policy, axis=1), k=5), tf.float32)
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

        accuracy_p1 = nan
        accuracy_p3 = nan
        accuracy_p5 = nan
        accuracy_v = nan

        while True:
            try:
                [policy_1, policy_3, policy_5, value_1] = sess.run([
                    policy_accuracy_1,
                    policy_accuracy_3,
                    policy_accuracy_5,
                    value_accuracy
                ])

                accuracy_p1 = policy_1 if isnan(accuracy_p1) else 0.97 * accuracy_p1 + 0.03 * policy_1
                accuracy_p3 = policy_3 if isnan(accuracy_p3) else 0.97 * accuracy_p3 + 0.03 * policy_3
                accuracy_p5 = policy_5 if isnan(accuracy_p5) else 0.97 * accuracy_p5 + 0.03 * policy_5
                accuracy_v = value_1 if isnan(accuracy_v) else 0.97 * accuracy_v + 0.03 * value_1
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

def dump(args):
    """
    Dump the given (or latest if none is given) checkpoint as a JSON file that
    is readable by dream-go
    """
    _tower = Tower()

    # restore only model variables
    saver_vars = tf.model_variables()
    saver = tf.train.Saver(saver_vars, keep_checkpoint_every_n_hours=2)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

        if args == []:
            latest_checkpoint = tf.train.latest_checkpoint('models/')
            if latest_checkpoint is not None:
                saver.restore(sess, latest_checkpoint)
        else:
            saver.restore(sess, args[0])

        # dump the variables to JSON in half precision in order to save
        # save disk space.
        import base64
        import json

        values = {}
        saver_vals = sess.run(saver_vars)

        for (var, value) in zip(saver_vars, saver_vals):
            if var in tf.get_collection(tf.GraphKeys.WEIGHTS):
                # tensorflow: [h, w, in, out]
                # cudnn:      [out, in, h, w]
                value = np.transpose(value, [3, 2, 0, 1])

            serialized = value.flatten().astype(np.float16).tostring()
            values[var.name] = base64.b85encode(serialized, pad=True).decode('ascii')

        json.dump(values, sys.stdout, sort_keys=True)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: bootstrap.py <data...>')
        quit()

    if sys.argv[1] == '--verify':
        verify(sys.argv[2:])
    if sys.argv[1] == '--dump':
        dump(sys.argv[2:])
    else:
        main(sys.argv[1:])
