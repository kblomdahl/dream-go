# Copyright (c) 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

def leela_zero(x, mode, params):
    data_type = tf.float16
    lz_init_op, num_blocks = read_lz_weights(
        params['lz_weights'],
        {
            'num_inputs': x.shape[3],
            'data_type': data_type
        }
    )

    def conv2d(x, name):
        y = tf.nn.conv2d(x, lz_init_op(f'{name}'), (1, 1, 1, 1), 'SAME', True, 'NHWC') + lz_init_op(f'{name}/offset')
        # z, _, _ = tf.nn.fused_batch_norm(
        #     y,
        #     tf.ones_like(lz_init_op(f'{name}/offset')),
        #     lz_init_op(f'{name}/offset'),
        #     lz_init_op(f'{name}/mean'),
        #     lz_init_op(f'{name}/variance'),
        #     epsilon=1e-5,
        #     data_format='NHWC',
        #     is_training=False
        # )
        # with tf.control_dependencies([tf.print(name, tf.math.reduce_mean(tf.math.reduce_mean(z, axis=[1, 2])), tf.math.reduce_mean(tf.math.reduce_variance(z, axis=[1, 2])))]):
        #     return tf.identity(z)
        with tf.control_dependencies([tf.print(name, tf.math.reduce_mean(tf.math.reduce_mean(y, axis=[1, 2])), tf.math.reduce_mean(tf.math.reduce_variance(y, axis=[1, 2])))]):
            return tf.identity(y)

    def dense(x, name):
        y = tf.matmul(x, lz_init_op(f'{name}')) + lz_init_op(f'{name}/offset')

        with tf.control_dependencies([tf.print(name, tf.math.reduce_mean(tf.math.reduce_mean(y, axis=[1])), tf.math.reduce_mean(tf.math.reduce_variance(y, axis=[1])))]):
            return tf.identity(y)

    x = tf.cast(x, data_type)

    with tf.variable_scope('lz', reuse=tf.AUTO_REUSE):
        # upsample
        with tf.name_scope('first_conv'):
            y = tf.nn.relu(conv2d(x, 'first_conv'))

        # residual blocks
        for i in range(num_blocks):
            original = tf.identity(y)
            with tf.name_scope(f'res_{i}_conv_1'):
                y = tf.nn.relu(conv2d(y, f'res_{i}_conv_1'))
            with tf.name_scope(f'res_{i}_conv_2'):
                y = tf.nn.relu(conv2d(y, f'res_{i}_conv_2') + original)

        # policy head
        with tf.name_scope('policy_head'):
            p = tf.nn.relu(conv2d(y, 'policy_head'))
            p = tf.transpose(p, [0, 3, 1, 2]) # NHWC -> NCHW
            p = tf.reshape(p, [-1, 722])
        with tf.name_scope('fc_1'):
            p = tf.nn.softmax(dense(p, 'w_fc_1'))

        # value head
        with tf.name_scope('value_head'):
            v = tf.nn.relu(conv2d(y, 'value_head'))
            v = tf.transpose(v, [0, 3, 1, 2]) # NHWC -> NCHW
            v = tf.reshape(v, [-1, 361])
        with tf.name_scope('fc_2'):
            v = tf.nn.relu(dense(v, 'w_fc_2'))
        with tf.name_scope('fc_3'):
            v = tf.nn.tanh(dense(v, 'w_fc_3'))

    return tf.stop_gradient(v), tf.stop_gradient(p)

def read_version(fh):
    line = fh.readline().strip()
    assert line == '1'

def read_numeric_line(fh):
    line = fh.readline().strip()
    return np.fromstring(line, 'f4', sep=' ')

def read_fold_bn_weights(fh, name, shape, out):
    weights = read_numeric_line(fh)
    beta = read_numeric_line(fh)
    mean = read_numeric_line(fh)
    variance = read_numeric_line(fh)
    std = np.sqrt(variance + 1e-5)

    out[f'{name}'] = np.transpose(np.reshape(weights, shape), [2, 3, 1, 0]) / std.reshape([1, 1, 1, shape[0]])
    out[f'{name}/offset'] = np.reshape(beta / std - mean / std, [shape[0]])
    out[f'{name}/mean'] = np.reshape(mean, [shape[0]])
    out[f'{name}/variance'] = np.reshape(variance, [shape[0]])

def read_linear_weights(fh, name, shape, out):
    out[name] = np.transpose(np.reshape(read_numeric_line(fh), shape), [1, 0])
    out[f'{name}/offset'] = np.reshape(read_numeric_line(fh), [shape[0]])

def count_lines(filename):
    with open(filename, 'r') as fh:
        return len([line for line in fh])

def calc_num_channels(filename, num_inputs):
    with open(filename, 'r') as fh:
        read_version(fh)
        return read_numeric_line(fh).size // (3 * 3 * num_inputs)

def read_lz_weights(filename, params):
    weights = {}
    data_type = params['data_type']
    num_inputs = params['num_inputs']
    num_channels = calc_num_channels(filename, num_inputs)
    num_blocks = (count_lines(filename) - (1 + 4 + 14)) // 8

    with open(filename, 'r') as fh:
        read_version(fh)
        read_fold_bn_weights(fh, 'first_conv', [num_channels, num_inputs, 3, 3], weights)

        for i in range(num_blocks):
            read_fold_bn_weights(fh, f'res_{i}_conv_1', [num_channels, num_channels, 3, 3], weights)
            read_fold_bn_weights(fh, f'res_{i}_conv_2', [num_channels, num_channels, 3, 3], weights)

        read_fold_bn_weights(fh, 'policy_head', [2, num_channels, 1, 1], weights)
        read_linear_weights(fh, 'w_fc_1', [362, 722], weights)

        read_fold_bn_weights(fh, 'value_head', [1, num_channels, 1, 1], weights)
        read_linear_weights(fh, 'w_fc_2', [256, 361], weights)
        read_linear_weights(fh, 'w_fc_3', [1, 256], weights)

    def lz_init_op(name):
        return tf.constant(weights[name].astype(data_type.as_numpy_dtype))

    return lz_init_op, num_blocks
