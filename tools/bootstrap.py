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

"""
Train the network weights to better predict the played moves and winner
in the given dataset.

Usage: ./bootstrap.py <dataset...>
"""

from datetime import datetime

import tensorflow as tf
import numpy as np

import sys

class BatchNorm:
    """
    Batch normalization layer.
    """

    def __init__(self, num_features, suffix=None):
        if suffix is None:
            suffix = ''

        ones_op = tf.ones_initializer()
        zeros_op = tf.zeros_initializer()

        self._scale = tf.get_variable('scale'+suffix, (num_features,), tf.float32, ones_op)
        self._offset = tf.get_variable('offset'+suffix, (num_features,), tf.float32, zeros_op)
        self._mean = tf.get_variable('mean'+suffix, (num_features,), tf.float32, zeros_op, trainable=False)
        self._variance = tf.get_variable('variance'+suffix, (num_features,), tf.float32, ones_op, trainable=False)

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._scale)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._offset)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._mean)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._variance)

    def forward(self, x, is_training=True):
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
                None,
                None,
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
        glorot_op = tf.glorot_uniform_initializer()

        self._conv_1 = tf.get_variable('weights_1', (3, 3, num_features, num_features), tf.float32, glorot_op)
        self._bn_1 = BatchNorm(num_features, '_1')
        self._conv_2 = tf.get_variable('weights_2', (3, 3, num_features, num_features), tf.float32, glorot_op)
        self._bn_2 = BatchNorm(num_features, '_2')

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._conv_1)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._conv_2)


    def forward(self, x, is_training=True):
        y = tf.nn.conv2d(x, self._conv_1, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn_1.forward(y, is_training)
        y = tf.nn.relu(y)
        y = tf.nn.conv2d(y, self._conv_2, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn_2.forward(y, is_training)

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
        glorot_op = tf.glorot_uniform_initializer()

        self._downsample = tf.get_variable('downsample', (1, 1, num_features, 1), tf.float32, glorot_op)
        self._bn = BatchNorm(1)
        self._weights_1 = tf.get_variable('weights_1', (361, 256), tf.float32, glorot_op)
        self._weights_2 = tf.get_variable('weights_2', (256, 1), tf.float32, glorot_op)

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._downsample)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._weights_1)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._weights_2)

    def forward(self, x, is_training=True):
        y = tf.nn.conv2d(x, self._downsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn.forward(y, is_training)
        y = tf.nn.relu(y)
        y = tf.reshape(y, (-1, 361))
        y = tf.matmul(y, self._weights_1)
        y = tf.nn.relu(y)
        y = tf.matmul(y, self._weights_2)
        y = tf.reshape(y, (-1,))

        return tf.nn.tanh(y)

class PolicyHead:
    """
    The policy head attached after the residual blocks as described by DeepMind:

    1. A convolution of 2 filters of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer that outputs a vector of size 192 + 1 = 362 corresponding to
       logit probabilities for all intersections and the pass move
    """

    def __init__(self, num_features):
        glorot_op = tf.glorot_uniform_initializer()

        self._downsample = tf.get_variable('downsample', (1, 1, num_features, 2), tf.float32, glorot_op)
        self._bn = BatchNorm(2)
        self._weights = tf.get_variable('weights', (722, 362), tf.float32, glorot_op)

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._downsample)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._weights)

    def forward(self, x, is_training=True):
        y = tf.nn.conv2d(x, self._downsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn.forward(y, is_training)
        y = tf.nn.relu(y)
        y = tf.reshape(y, (-1, 722))
        y = tf.matmul(y, self._weights)

        return y

class Tower:
    """
    The full neural network used to predict the value and policy tensors for a mini-batch of board positions.
    """

    def __init__(self, num_features=256):
        glorot_op = tf.glorot_uniform_initializer()

        with tf.variable_scope('01_upsample'):
            self._upsample = tf.get_variable('weights', (3, 3, 14, num_features), tf.float32, glorot_op)
            self._bn = BatchNorm(num_features)

        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, self._upsample)

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

    def forward(self, x, is_training=True):
        y = tf.nn.conv2d(x, self._upsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn.forward(y, is_training)
        y = tf.nn.relu(y)

        for resb in self._residuals:
            y = resb.forward(y)

        p = self._policy.forward(y)
        v = self._value.forward(y)

        return v, p

if __name__ == '__main__':
    if len(sys.argv) >= 1:
        print('Usage: bootstrap.py <data...>')
        quit()

    dataset = tf.data.FixedLengthRecordDataset(sys.argv[1:], 10834)
    dataset = dataset.map(lambda x: tf.cast(tf.decode_raw(x, tf.half), tf.float32))
    dataset = dataset.map(lambda x: tf.split(x, (5054, 1, 362)))
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(256)

    iterator = dataset.make_initializable_iterator()

    #
    tower = Tower()

    with tf.device('cpu:0'):
        global_step = tf.train.create_global_step()
        epoch = tf.get_variable('epoch', (), tf.int64, tf.zeros_initializer(), trainable=False)
        epoch_op = tf.assign_add(epoch, 1)

    features, value, policy = iterator.get_next()
    features = tf.reshape(features, (-1, 14, 19, 19))
    value_hat, policy_hat = tower.forward(features)

    #
    policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=policy_hat))
    value_loss = tf.reduce_mean(tf.squared_difference(value, value_hat))
    reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    loss = policy_loss + 1e-2 * value_loss + 1e-4 * reg_loss

    tf.summary.scalar('loss/policy', policy_loss)
    tf.summary.scalar('loss/value', value_loss)
    tf.summary.scalar('loss/regularization', reg_loss)
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.MomentumOptimizer(1e-2, 0.9)
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

    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('epoch', epoch)

    # operations
    summary_writer = tf.summary.FileWriter('logs/' + datetime.now().strftime('%Y%m%d.%H%M') + '/')
    summary_op = tf.summary.merge_all()
    update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    restore_op = tf.no_op()  # fixme
    save_op = tf.no_op()  # fixme

    with tf.Session() as sess:
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
        sess.graph.finalize()

        # restore model from checkpoint
        sess.run(restore_op)

        while True:
            sess.run(iterator.initializer)
            epoch_hat = sess.run(epoch)

            try:
                while True:
                    global_step_hat, _, _ = sess.run([global_step, optimizer_op, update_op])

                    if global_step_hat > 0 and global_step_hat % 10 == 0:
                        summary_hat = sess.run(summary_op)
                        summary_writer.add_summary(summary_hat, global_step_hat)
                    if global_step_hat > 0 and global_step_hat % 1000 == 0:
                        sess.run(save_op)
            except KeyboardInterrupt:
                break  # quit
            except:
                sess.run([epoch_op, save_op])

        # save the model
        sess.run(save_op)

