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
import gzip

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
        return tf.nn.conv2d(x, lz_init_op(name), (1, 1, 1, 1), 'SAME', True, 'NHWC') + lz_init_op(f'{name}/offset')

    def dense(x, name):
        return tf.matmul(x, lz_init_op(name)) + lz_init_op(f'{name}/offset')

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

    return tf.stop_gradient(v), tf.stop_gradient(p), tf.stop_gradient(y)

class LzWeightsParser:
    def __init__(self, filename, params):
        self.filename = filename
        self.params = params
        self.out = {}

    def __enter__(self):
        self.fh = self._open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.fh.close()
        self.fh = None

    @property
    def num_blocks(self):
        with self._open() as fh:
            count = len([line for line in fh])

        return (count - (1 + 4 + 14)) // 8

    @property
    def num_channels(self):
        num_inputs = self.params['num_inputs']

        with self._open() as fh:
            self._read_version(fh)
            return self._read_numeric_line(fh).size // (3 * 3 * num_inputs)

    def _open(self):
        if self.filename.endswith('.gz'):
            return gzip.open(self.filename, 'rt')
        else:
            return open(self.filename, 'r')

    def to_dict(self):
        return self.out

    def _read_version(self, fh):
        line = fh.readline().strip()
        assert line == '1', f'version should be 1 -- {line}'

    def read_version(self):
        self._read_version(self.fh)

    def _read_numeric_line(self, fh):
        line = fh.readline().strip()
        return np.fromstring(line, 'f4', sep=' ')

    def read_numeric_line(self):
        return self._read_numeric_line(self.fh)

    def read_fold_bn_weights(self, name, shape):
        weights = self.read_numeric_line()
        beta = self.read_numeric_line()
        mean = self.read_numeric_line()
        variance = self.read_numeric_line()
        std = np.sqrt(variance + 1e-5)

        self.out[f'{name}'] = np.transpose(np.reshape(weights, shape), [2, 3, 1, 0]) / std.reshape([1, 1, 1, shape[0]])
        self.out[f'{name}/offset'] = np.reshape(beta / std - mean / std, [shape[0]])
        self.out[f'{name}/mean'] = np.reshape(mean, [shape[0]])
        self.out[f'{name}/variance'] = np.reshape(variance, [shape[0]])

    def read_linear_weights(self, name, shape):
        weights = self.read_numeric_line()
        beta = self.read_numeric_line()

        self.out[name] = np.transpose(np.reshape(weights, shape), [1, 0])
        self.out[f'{name}/offset'] = np.reshape(beta, [shape[0]])


def read_lz_weights(filename, params):
    data_type = params['data_type']

    with LzWeightsParser(filename, params) as p:
        num_blocks = p.num_blocks
        num_channels = p.num_channels
        num_inputs = params['num_inputs']

        p.read_version()
        p.read_fold_bn_weights('first_conv', [num_channels, num_inputs, 3, 3])

        for i in range(p.num_blocks):
            p.read_fold_bn_weights(f'res_{i}_conv_1', [num_channels, num_channels, 3, 3])
            p.read_fold_bn_weights(f'res_{i}_conv_2', [num_channels, num_channels, 3, 3])

        p.read_fold_bn_weights('policy_head', [2, num_channels, 1, 1])
        p.read_linear_weights('w_fc_1', [362, 722])

        p.read_fold_bn_weights('value_head', [1, num_channels, 1, 1])
        p.read_linear_weights('w_fc_2', [256, 361])
        p.read_linear_weights('w_fc_3', [1, 256])

        #
        weights = p.to_dict()

    def lz_init_op(name):
        return tf.constant(weights[name].astype(data_type.as_numpy_dtype))

    return lz_init_op, num_blocks
