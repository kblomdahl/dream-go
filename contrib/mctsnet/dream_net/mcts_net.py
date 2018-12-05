import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

from dream_net.background_service import BackgroundService

# -------- Private methods --------

NUM_CHANNELS = 96

def orthogonal_initializer(shape):
    """ Returns an orthogonal initializer that use QR-factorization to find
    the orthogonal basis of a random matrix. This differs from the Tensorflow
    implementation in that it checks for singular matrices, which is a
    problem when generating small matrices. """

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

class Conv2D:
    def __init__(self, size, num_inputs, num_outputs):
        total_inputs = size * size * num_inputs

        self.scale = tfe.Variable(
            np.ones([num_outputs], 'f4'),
            trainable=True,
            dtype=tf.float32
        )
        self.offset = tfe.Variable(
            np.zeros([num_outputs]),
            trainable=True,
            dtype=tf.float32
        )
        self.weights = tfe.Variable(
            orthogonal_initializer([size, size, num_inputs, num_outputs]) / total_inputs,  # initial value
            trainable=True,
            dtype=tf.float32
        )

    def all_variables(self, prefix, acc):
        acc[prefix + '/scale'] = self.scale
        acc[prefix + '/offset'] = self.offset
        acc[prefix + '/weights'] = self.weights

    def forward(self, x):
        y = tf.nn.conv2d(x, self.weights, (1, 1, 1, 1), 'SAME', True, 'NCHW')

        # apply batch normalization
        y, y_mean, y_var = tf.nn.fused_batch_norm(
            y,
            self.scale,
            self.offset,
            epsilon=0.0001,
            data_format='NCHW',
            is_training=True
        )

        # TODO Accumulate y_mean, and y_var using a moving average

        return y

class Linear:
    def __init__(self, num_inputs, num_outputs, bias_scale=0.0):
        total_inputs = num_inputs

        self.offset = tfe.Variable(
            bias_scale * np.random.uniform(0.0, 1.0, [1, num_outputs]),
            trainable=True,
            dtype=tf.float32
        )
        self.weights = tfe.Variable(
            orthogonal_initializer([num_inputs, num_outputs]) / total_inputs,  # initial value
            trainable=True,
            dtype=tf.float32
        )

    def all_variables(self, prefix, acc):
        acc[prefix + '/offset'] = self.offset
        acc[prefix + '/weights'] = self.weights

    def forward(self, x):
        return tf.matmul(x, self.weights) + self.offset


class ResidualBlock:
    def __init__(self, size, num_channels):
        self.conv_1 = Conv2D(size, num_channels, num_channels)
        self.conv_2 = Conv2D(size, num_channels, num_channels)

    def all_variables(self, prefix, acc):
        self.conv_1.all_variables(prefix + '/conv_1', acc)
        self.conv_2.all_variables(prefix + '/conv_2', acc)

    def forward(self, x):
        y = tf.nn.relu(self.conv_1.forward(x))

        return tf.nn.relu(x + self.conv_2.forward(y))


class Embedding:
    __NETWORK = None

    def __init__(self):
        with tf.variable_scope('embedding'):
            self.upscale = Conv2D(3, 4, NUM_CHANNELS)
            self.downscale = Conv2D(1, NUM_CHANNELS, 2)
            self.residual = {}

            for i in range(9):
                self.residual[i] = ResidualBlock(3, NUM_CHANNELS)

    def all_variables(self, prefix, acc):
        self.upscale.all_variables(prefix + '/01_upsample', acc)
        for i, residual in self.residual.items():
            residual.all_variables(prefix + '/{:02d}_residual'.format(2 + i), acc)
        self.downscale.all_variables(prefix + '/11_downsample', acc)

    def forward(self, x):
        with tf.name_scope('embedding'):
            y = tf.nn.relu(self.upscale.forward(x))

            for i in range(9):
                y = self.residual[i].forward(y)

            y = tf.nn.tanh(self.downscale.forward(y))
            return tf.reshape(y, (-1, 722))

    @staticmethod
    def get_instance():
        if not Embedding.__NETWORK:
            Embedding.__NETWORK = Embedding()
        return Embedding.__NETWORK


class Backup:
    """ Backup Network, this network is used to combine the parent statistics
    with the statistics of its child. It uses a Gated Recurrent Unit [1]
    architecture.

    [1] "Learning Phrase Representations using RNN Encoder-Decoder for
        Statistical Machine Translation", Kyunghyun Cho, Bart van Merrienboer,
        Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk,
        Yoshua Bengio, https://arxiv.org/abs/1406.1078
    """

    __NETWORK = None

    def __init__(self):
        with tf.variable_scope('backup'):
            self.input_w = Linear(722, 722, bias_scale=1.0)
            self.input_r = Linear(722, 722, bias_scale=1.0)

            self.reset_w = Linear(722, 722, bias_scale=1.0)
            self.reset_r = Linear(722, 722, bias_scale=1.0)

            self.update_w = Linear(722, 722, bias_scale=0.0)
            self.update_r = Linear(722, 722, bias_scale=0.0)

    def all_variables(self, prefix, acc):
        self.input_w.all_variables(prefix + '/input_w', acc)
        self.input_r.all_variables(prefix + '/input_r', acc)
        self.reset_r.all_variables(prefix + '/reset_r', acc)
        self.reset_w.all_variables(prefix + '/reset_w', acc)
        self.update_w.all_variables(prefix + '/update_w', acc)
        self.update_r.all_variables(prefix + '/update_r', acc)

    def forward(self, h_parent, x):
        with tf.name_scope('backup'):
            input = tf.nn.sigmoid(self.input_w.forward(x) + self.input_r.forward(h_parent))
            reset = tf.nn.sigmoid(self.reset_w.forward(x) + self.reset_r.forward(h_parent))

            h_hat = tf.nn.tanh(self.update_w.forward(x) + reset * self.update_r.forward(h_parent))

            return (1.0 - input) * h_hat + input * h_parent

    @staticmethod
    def get_instance():
        if not Backup.__NETWORK:
            Backup.__NETWORK = Backup()
        return Backup.__NETWORK


class Readout:
    __NETWORK = None

    def __init__(self):
        with tf.variable_scope('readout'):
            with tf.variable_scope('policy'):
                self.policy = Linear(722, 362)
            with tf.variable_scope('value'):
                self.value_1 = Linear(722, 256)
                self.value_2 = Linear(256, 1)

    def all_variables(self, prefix, acc):
        self.policy.all_variables(prefix + '/01p_linear', acc)
        self.value_1.all_variables(prefix + '/01v_linear', acc)
        self.value_2.all_variables(prefix + '/02v_linear', acc)

    def forward(self, x):
        with tf.name_scope('readout/policy'):
            p = self.policy.forward(x)

        with tf.name_scope('readout/value'):
            v = tf.nn.relu(self.value_1.forward(x))
            v = tf.nn.tanh(self.value_2.forward(v))

        return p, v

    @staticmethod
    def get_instance():
        if not Readout.__NETWORK:
            Readout.__NETWORK = Readout()
        return Readout.__NETWORK


class Policy:
    __NETWORK = None

    def __init__(self):
        with tf.variable_scope('policy'):
            self.linear = Linear(722, 362)

    def all_variables(self, prefix, acc):
        self.linear.all_variables(prefix + '/linear', acc)

    def forward(self, x):
        y = self.linear.forward(x)

        return tf.nn.softmax(y)

    @staticmethod
    def get_instance():
        if not Policy.__NETWORK:
            Policy.__NETWORK = Policy()
        return Policy.__NETWORK


# -------- Public methods --------

def all_variables():
    acc = {}

    Embedding.get_instance().all_variables('embedding', acc)
    Backup.get_instance().all_variables('backup', acc)
    Readout.get_instance().all_variables('readout', acc)
    Policy.get_instance().all_variables('policy', acc)

    return acc

def embed(board, color):
    x = board.get_features(color).reshape([1, -1, 19, 19])

    return Embedding.get_instance().forward(x)

def backup(h_parent, h):
    return Backup.get_instance().forward(h_parent, h)

def readout(h):
    return Readout.get_instance().forward(h)

def policy(h):
    return Policy.get_instance().forward(h)
