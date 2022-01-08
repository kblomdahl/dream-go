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

import argparse
from datetime import datetime
import os

from tensorboard.plugins.hparams import api as hp

class Config:
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.IntInterval(1, 8196))
    HP_NUM_UNROLLS = hp.HParam('num_unrolls', hp.IntInterval(0, 32))
    HP_NUM_BLOCKS = hp.HParam('num_blocks', hp.IntInterval(0, 80))
    HP_NUM_DYNAMICS_BLOCKS = hp.HParam('num_dynamics_blocks', hp.IntInterval(0, 80))
    HP_NUM_CHANNELS = hp.HParam('num_channels', hp.IntInterval(1, 2048))
    HP_NUM_DYNAMICS_CHANNELS = hp.HParam('num_dynamics_channels', hp.IntInterval(1, 2048))
    HP_EMBEDDINGS_SIZE = hp.HParam('embeddings_size', hp.IntInterval(1, 2048))
    HP_POLICY_COEF = hp.HParam('policy_coefficient', hp.RealInterval(0.0, 1.0))
    HP_VALUE_COEF = hp.HParam('value_coefficient', hp.RealInterval(0.0, 1.0))
    HP_OWNERSHIP_COEF = hp.HParam('ownership_coefficient', hp.RealInterval(0.0, 1.0))
    HP_SIMILARITY_COEF = hp.HParam('similarity_coefficient', hp.RealInterval(0.0, 1.0))
    HP_DISCOUNT_FACTOR = hp.HParam('discount_factor', hp.RealInterval(0.0, 1.0))
    HP_WEIGHT_DECAY = hp.HParam('weight_decay', hp.RealInterval(0.0, 1.0))
    HP_LABEL_SMOOTHING = hp.HParam('label_smoothing', hp.RealInterval(0.0, 1.0))
    HP_INITIAL_LR = hp.HParam('initial_lr', hp.RealInterval(0.0, 1.0))
    HP_MAX_LR = hp.HParam('max_lr', hp.RealInterval(0.0, 1.0))
    HP_LR_NUM_WARMUP_STEPS = hp.HParam('lr_num_warmup_steps', hp.RealInterval(0.0, 1.0))
    HP_LR_DECAY_STEPS = hp.HParam('lr_decay_steps', hp.IntInterval(0, 100000))
    HP_LR_DECAY_RATE = hp.HParam('lr_decay_rate', hp.RealInterval(0.0, 1.0))
    HP_ES_NUM_WARMUP_STEPS = hp.HParam('es_num_warmup_steps', hp.IntInterval(0, 100000))
    HP_ES_NUM_SAMPLES = hp.HParam('es_num_samples', hp.IntInterval(3, 100000))
    HP_ES_MAX_SLOPE = hp.HParam('es_max_slope', hp.RealInterval(-1.0, 0.0))
    HP_CLIPNORM = hp.HParam('clipnorm', hp.RealInterval(0.0, 1000.0))
    HP_EPOCHS = hp.HParam('epochs', hp.IntInterval(0, 100000))
    HP_LZ_WEIGHTS = hp.HParam('lz_weights')
    HP_FILES = hp.HParam('files')
    HP_NAME = hp.HParam('name')
    HP_WARM_START = hp.HParam('warm_start')

    def __init__(self, args=None, exit_on_error=True):
        self.args = self.parse_args(args, exit_on_error=exit_on_error)
        self.hparams = {
            self.HP_BATCH_SIZE: self.args.batch_size,
            self.HP_NUM_UNROLLS: self.args.num_unrolls,
            self.HP_NUM_BLOCKS: self.args.num_blocks,
            self.HP_NUM_DYNAMICS_BLOCKS: self.args.num_dynamics_blocks,
            self.HP_NUM_CHANNELS: self.args.num_channels,
            self.HP_NUM_DYNAMICS_CHANNELS: self.args.num_dynamics_channels,
            self.HP_EMBEDDINGS_SIZE: self.args.embeddings_size,
            self.HP_POLICY_COEF: self.args.policy_coefficient,
            self.HP_VALUE_COEF: self.args.value_coefficient,
            self.HP_OWNERSHIP_COEF: self.args.ownership_coefficient,
            self.HP_SIMILARITY_COEF: self.args.similarity_coefficient,
            self.HP_DISCOUNT_FACTOR: self.args.discount_factor,
            self.HP_WEIGHT_DECAY: self.args.weight_decay,
            self.HP_LABEL_SMOOTHING: self.args.label_smoothing,
            self.HP_INITIAL_LR: self.args.initial_learning_rate,
            self.HP_MAX_LR: self.args.max_learning_rate,
            self.HP_LR_NUM_WARMUP_STEPS: self.args.num_warmup_steps,
            self.HP_LR_DECAY_STEPS: self.args.num_decay_steps,
            self.HP_LR_DECAY_RATE: self.args.decay_rate,
            self.HP_ES_NUM_WARMUP_STEPS: self.args.num_es_warmup_steps,
            self.HP_ES_NUM_SAMPLES: self.args.num_es_samples,
            self.HP_ES_MAX_SLOPE: self.args.max_es_slope,
            self.HP_CLIPNORM: self.args.clipnorm,
            self.HP_LZ_WEIGHTS: self.args.lz_weights or '',
            self.HP_FILES: ','.join(self.args.files) or '',
            self.HP_NAME: self.args.name or '',
            self.HP_WARM_START: self.args.warm_start or '',
            self.HP_EPOCHS: self.args.epochs
        }

    def parse_args(self, args, exit_on_error=True):
        parser = argparse.ArgumentParser(
            prog='dream_tf',
            description='Neural network optimizer for Dream Go.',
            #exit_on_error=exit_on_error  # wait until python 3.9
        )

        parser.add_argument('files', nargs=argparse.REMAINDER, help='The binary features files.')

        opt_group = parser.add_argument_group(title='optional configuration')
        opt_group.add_argument('--model', nargs='?', help='the directory that contains the model')
        opt_group.add_argument('--name', nargs='?', help='the name of this session')
        opt_group.add_argument('--warm-start', nargs='?', help='the model to warm-start from')
        opt_group.add_argument('--lz-weights', nargs='?', help='leela-zero weights to use for semi-supervised learning')
        opt_group.add_argument('--run-eagerly', action='store_true', help='run the model in eager mode')

        opt_group = parser.add_argument_group(title='model configuration')
        opt_group.add_argument('--batch-size', default=2816, nargs='?', type=int, metavar='N', help='the number of examples per mini-batch')
        opt_group.add_argument('--num-unrolls', default=8, nargs='?', type=int, metavar='N', help='the number of board states per example')
        opt_group.add_argument('--num-channels', default=128, nargs='?', type=int, metavar='N', help='the number of channels per residual block')
        opt_group.add_argument('--num-blocks', default=9, nargs='?', type=int, metavar='N', help='the number of residual blocks in the representation')
        opt_group.add_argument('--num-dynamics-blocks', default=6, nargs='?', type=int, metavar='N', help='the number of residual blocks in the dynamics')
        opt_group.add_argument('--num-dynamics-channels', default=64, nargs='?', type=int, metavar='N', help='the number of channels in the dynamics')
        opt_group.add_argument('--embeddings-size', default=722, nargs='?', type=int, metavar='N', help='the number of values in the state embedding')
        opt_group.add_argument('--policy-coefficient', default=1.0, nargs='?', type=float, metavar='N', help='the policy coefficient in the total loss')
        opt_group.add_argument('--value-coefficient', default=1.0, nargs='?', type=float, metavar='N', help='the value coefficient in the total loss')
        opt_group.add_argument('--ownership-coefficient', default=0.1, nargs='?', type=float, metavar='N', help='the ownership coefficient in the total loss')
        opt_group.add_argument('--similarity-coefficient', default=0.1, nargs='?', type=float, metavar='N', help='the similarity coefficient in the total loss')
        opt_group.add_argument('--discount-factor', default=0.97, nargs='?', type=float, metavar='N', help='the lambda discount factor')
        opt_group.add_argument('--weight-decay', default=1e-6, nargs='?', type=float, metavar='N', help='the weight decay')
        opt_group.add_argument('--label-smoothing', default=0.1, nargs='?', type=float, metavar='N', help='the label smoothing')
        opt_group.add_argument('--initial-learning-rate', default=1e-4, nargs='?', type=float, metavar='N', help='the initial learning rate')
        opt_group.add_argument('--max-learning-rate', default=3e-4, nargs='?', type=float, metavar='N', help='the maximum learning rate after warmup')
        opt_group.add_argument('--num-warmup-steps', default=2500, nargs='?', type=int, metavar='N', help='the number of warmup steps')
        opt_group.add_argument('--num-decay-steps', default=240, nargs='?', type=int, metavar='N', help='the number of steps per learning rate decay')
        opt_group.add_argument('--decay-rate', default=0.97, nargs='?', type=float, metavar='N', help='the learning rate decay')
        opt_group.add_argument('--num-es-warmup-steps', default=5000, nargs='?', type=int, metavar='N', help='the number of steps to ignore in early stopping')
        opt_group.add_argument('--num-es-samples', default=50, nargs='?', type=int, metavar='N', help='the number of values to take into account during early stopping')
        opt_group.add_argument('--max-es-slope', default=-1e-7, nargs='?', type=float, metavar='N', help='the minimum slope allowed before early stopping')
        opt_group.add_argument('--clipnorm', default=5.0, nargs='?', type=float, metavar='N', help='the gradient scaling norm')
        opt_group.add_argument('--epochs', default=500, nargs='?', type=int, metavar='N', help='the maximum epochs to train for')

        op_group = parser.add_mutually_exclusive_group(required=True)
        op_group.add_argument('--start', action='store_true', help='start training of a new model')
        op_group.add_argument('--resume', action='store_true', help='resume training of a model')
        op_group.add_argument('--verify', action='store_true', help='evaluate the accuracy of a model')
        op_group.add_argument('--dump', action='store_true', help='print the weights of a model to standard output')

        return parser.parse_args(args)

    def has_model(self):
        return self.args.model is not None

    def get_model_dir(self, base_model_dir='models', default=None):
        model_dir = self.args.model
        if not model_dir and default is None:
            model_dir = base_model_dir + '/' + datetime.now().strftime('%Y%m%d.%H%M')

            if self.args.name:
                model_dir += '-' + self.args.name + '/'
            else:
                model_dir += '/'

        return model_dir or default

    def is_start(self):
        return self.args.start

    def is_resume(self):
        return self.args.resume

    def is_verify(self):
        return self.args.verify

    def is_dump(self):
        return self.args.dump

    @property
    def run_eagerly(self):
        return self.args.run_eagerly

    @property
    def files(self):
        return self.args.files

    @property
    def warm_start(self):
        return self.hparams[self.HP_WARM_START]

    @property
    def lz_weights(self):
        return self.hparams[self.HP_LZ_WEIGHTS]

    @property
    def batch_size(self):
        return self.hparams[self.HP_BATCH_SIZE]

    @property
    def num_unrolls(self):
        return self.hparams[self.HP_NUM_UNROLLS]

    @property
    def num_blocks(self):
        return self.hparams[self.HP_NUM_BLOCKS]

    @property
    def num_dynamics_blocks(self):
        return self.hparams[self.HP_NUM_DYNAMICS_BLOCKS]

    @property
    def num_channels(self):
        return self.hparams[self.HP_NUM_CHANNELS]

    @property
    def num_dynamics_channels(self):
        return self.hparams[self.HP_NUM_DYNAMICS_CHANNELS]

    @property
    def embeddings_size(self):
        return self.hparams[self.HP_EMBEDDINGS_SIZE]

    @property
    def policy_coefficient(self):
        return self.hparams[self.HP_POLICY_COEF]

    @property
    def value_coefficient(self):
        return self.hparams[self.HP_VALUE_COEF]

    @property
    def ownership_coefficient(self):
        return self.hparams[self.HP_OWNERSHIP_COEF]

    @property
    def similarity_coefficient(self):
        return self.hparams[self.HP_SIMILARITY_COEF]

    @property
    def discount_factor(self):
        return self.hparams[self.HP_DISCOUNT_FACTOR]

    @property
    def weight_decay(self):
        return self.hparams[self.HP_WEIGHT_DECAY]

    @property
    def label_smoothing(self):
        return self.hparams[self.HP_LABEL_SMOOTHING]

    @property
    def initial_learning_rate(self):
        return self.hparams[self.HP_INITIAL_LR]

    @property
    def max_learning_rate(self):
        return self.hparams[self.HP_MAX_LR]

    @property
    def num_warmup_steps(self):
        return self.hparams[self.HP_LR_NUM_WARMUP_STEPS]

    @property
    def num_decay_steps(self):
        return self.hparams[self.HP_LR_DECAY_STEPS]

    @property
    def decay_rate(self):
        return self.hparams[self.HP_LR_DECAY_RATE]

    @property
    def num_es_warmup_steps(self):
        return self.hparams[self.HP_ES_NUM_WARMUP_STEPS]

    @property
    def num_es_samples(self):
        return self.hparams[self.HP_ES_NUM_SAMPLES]

    @property
    def max_es_slope(self):
        return self.hparams[self.HP_ES_MAX_SLOPE]

    @property
    def clipnorm(self):
        return self.hparams[self.HP_CLIPNORM]

    @property
    def epochs(self):
        return self.hparams[self.HP_EPOCHS]

def most_recent_model():
    """ Returns the directory in `models/` that is the most recent """

    all_models = ['models/' + m for m in os.listdir('models/')]

    return max(
        [m for m in all_models if os.path.isdir(m)],
        key=os.path.getmtime
    )
