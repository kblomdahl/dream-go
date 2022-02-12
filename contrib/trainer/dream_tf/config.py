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
from functools import cached_property
import os

from .model_config import ModelConfig
from tensorboard.plugins.hparams import api as hp

_HP_NAME = hp.HParam('name')
_HP_WARM_START = hp.HParam('warm_start')

class Config:
    def __init__(self, args=None, exit_on_error=True):
        self.args = self.parse_args(args, exit_on_error=exit_on_error)
        self.hparams = {
            _HP_NAME: self.args.name or '',
            _HP_WARM_START: self.args.warm_start or ''
        }

    def parse_args(self, args, exit_on_error=True):
        parser = argparse.ArgumentParser(
            prog='dream_tf',
            description='Neural network optimizer for Dream Go.',
            #exit_on_error=exit_on_error  # wait until python 3.9
        )

        opt_group = parser.add_argument_group(title='optional configuration')
        opt_group.add_argument('--config', type=argparse.FileType('r'), help='the model configuration file')
        opt_group.add_argument('--model', help='the directory that contains the model')
        opt_group.add_argument('--name', help='the name of this session')
        opt_group.add_argument('--warm-start', help='the model to warm-start from')

        op_group = parser.add_mutually_exclusive_group(required=True)
        op_group.add_argument('--start', action='store_true', help='start training of a new model')
        op_group.add_argument('--resume', action='store_true', help='resume training of a model')
        op_group.add_argument('--verify', action='store_true', help='evaluate the accuracy of a model')
        op_group.add_argument('--dump', action='store_true', help='print the weights of a model to standard output')

        return parser.parse_args(args)

    @property
    def name(self):
        return self.hparams[_HP_NAME]

    @property
    def warm_start(self):
        return self.hparams[_HP_WARM_START]

    @cached_property
    def model_config(self):
        if self.args.config:
            return ModelConfig(self.args.config)
        elif self.has_model():
            model_dir = self.get_model_dir(default=most_recent_model())

            return ModelConfig(f'{model_dir}/config.json')
        else:
            return ModelConfig()

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

def most_recent_model():
    """ Returns the directory in `models/` that is the most recent """

    all_models = ['models/' + m for m in os.listdir('models/')]

    return max(
        [m for m in all_models if os.path.isdir(m)],
        key=os.path.getmtime
    )
