# Copyright (c) 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

import json

from tensorboard.plugins.hparams import api as hp

_HP_BATCH_SIZE = hp.HParam('batch_size', hp.IntInterval(1, 8196))
_HP_NUM_UNROLLS = hp.HParam('num_unrolls', hp.IntInterval(0, 32))
_HP_LZ_WEIGHTS = hp.HParam('lz_weights')
_HP_DATA = hp.HParam('data')

_HP_EMBEDDINGS_SIZE = hp.HParam('model.embeddings_size', hp.IntInterval(1, 2048))
_HP_NUM_REPR_BLOCKS = hp.HParam('model.repr.num_blocks', hp.IntInterval(0, 80))
_HP_NUM_REPR_CHANNELS = hp.HParam('model.repr.num_channels', hp.IntInterval(1, 2048))
_HP_NUM_DYN_BLOCKS = hp.HParam('model.dyn.num_blocks', hp.IntInterval(0, 80))
_HP_NUM_DYN_CHANNELS = hp.HParam('model.dyn.num_channels', hp.IntInterval(1, 2048))

_HP_POLICY_COEF = hp.HParam('loss.coefficients.policy', hp.RealInterval(0.0, 1.0))
_HP_VALUE_COEF = hp.HParam('loss.coefficients.value', hp.RealInterval(0.0, 1.0))
_HP_OWNERSHIP_COEF = hp.HParam('loss.coefficients.ownership', hp.RealInterval(0.0, 1.0))
_HP_SIMILARITY_COEF = hp.HParam('loss.coefficients.similarity', hp.RealInterval(0.0, 1.0))
_HP_LABEL_SMOOTHING = hp.HParam('loss.label_smoothing', hp.RealInterval(0.0, 1.0))
_HP_DISCOUNT_FACTOR = hp.HParam('loss.discount_factor', hp.RealInterval(0.0, 1.0))

_HP_LEARNING_RATE = hp.HParam('optimizer.learning_rate', hp.RealInterval(0.0, 1.0))
_HP_DECAY_RATE = hp.HParam('optimizer.decay_rate', hp.RealInterval(0.0, 1.0))
_HP_WEIGHT_DECAY = hp.HParam('optimizer.weight_decay', hp.RealInterval(0.0, 1.0))
_HP_CLIPNORM = hp.HParam('optimizer.clipnorm', hp.RealInterval(0.0, 1000.0))
_HP_EPOCHS = hp.HParam('optimizer.epochs', hp.IntInterval(0, 100000))

_HP_NUM_ES_WARMUP_STEPS = hp.HParam('early_stopping.num_warmup_steps', hp.IntInterval(0, 100000))
_HP_NUM_ES_SAMPLES = hp.HParam('early_stopping.num_samples', hp.IntInterval(3, 100000))
_HP_MAX_ES_SLOPE = hp.HParam('early_stopping.max_slope', hp.RealInterval(-1.0, 0.0))

class ModelConfig:
    def __init__(self, fp=None):
        if fp is not None:
            self._parse(json.load(fp))
        else:
            self._parse({})

    def _dig(self, *args):
        content = self.content

        for i in range(len(args)):
            if type(content) is not dict:
                return None
            elif args[i] not in content:
                return None
            else:
                content = content[args[i]]

        return content

    def _parse(self, content: dict):
        self.content = content
        self.hparams = {
            _HP_BATCH_SIZE: self._dig('batch_size') or 2048,
            _HP_NUM_UNROLLS: self._dig('num_unrolls') or 8,
            _HP_LZ_WEIGHTS: self._dig('lz_weights'),
            _HP_DATA: self._dig('data') or [],

            _HP_EMBEDDINGS_SIZE: self._dig('model', 'embeddings_size') or 722,
            _HP_NUM_REPR_BLOCKS: self._dig('model', 'repr', 'num_blocks') or 6,
            _HP_NUM_REPR_CHANNELS: self._dig('model', 'repr', 'num_channels') or 96,
            _HP_NUM_DYN_BLOCKS: self._dig('model', 'dyn', 'num_blocks') or 6,
            _HP_NUM_DYN_CHANNELS: self._dig('model', 'dyn', 'num_channels') or 64,

            _HP_POLICY_COEF: self._dig('loss', 'coefficients', 'policy') or 1.0,
            _HP_VALUE_COEF: self._dig('loss', 'coefficients', 'value') or 1.0,
            _HP_OWNERSHIP_COEF: self._dig('loss', 'coefficients', 'ownership') or 0.1,
            _HP_SIMILARITY_COEF: self._dig('loss', 'coefficients', 'similarity') or 0.1,
            _HP_LABEL_SMOOTHING: self._dig('loss', 'label_smoothing') or 0.2,
            _HP_DISCOUNT_FACTOR: self._dig('loss', 'discount_factor') or 0.97,

            _HP_LEARNING_RATE: self._dig('optimizer', 'learning_rate') or 3e-3,
            _HP_DECAY_RATE: self._dig('optimizer', 'decay_rate') or 0.96,
            _HP_WEIGHT_DECAY: self._dig('optimizer', 'weight_decay') or 1e-5,
            _HP_CLIPNORM: self._dig('optimizer', 'clipnorm') or 5.0,
            _HP_EPOCHS: self._dig('optimizer', 'epochs') or 100,

            _HP_NUM_ES_WARMUP_STEPS: self._dig('early_stopping', 'num_warmup_steps') or 1000,
            _HP_NUM_ES_SAMPLES: self._dig('early_stopping', 'samples') or 50,
            _HP_MAX_ES_SLOPE: self._dig('early_stopping', 'max_slope') or 1e-7,
        }

    @property
    def batch_size(self):
        return self.hparams[_HP_BATCH_SIZE]

    @property
    def num_unrolls(self):
        return self.hparams[_HP_NUM_UNROLLS]

    @property
    def lz_weights(self):
        return self.hparams[_HP_LZ_WEIGHTS]

    @property
    def data(self):
        return self.hparams[_HP_DATA]

    @property
    def embeddings_size(self):
        return self.hparams[_HP_EMBEDDINGS_SIZE]

    @property
    def num_repr_blocks(self):
        return self.hparams[_HP_NUM_REPR_BLOCKS]

    @property
    def num_repr_channels(self):
        return self.hparams[_HP_NUM_REPR_CHANNELS]

    @property
    def num_dyn_blocks(self):
        return self.hparams[_HP_NUM_DYN_BLOCKS]

    @property
    def num_dyn_channels(self):
        return self.hparams[_HP_NUM_DYN_CHANNELS]

    @property
    def policy_coefficient(self):
        return self.hparams[_HP_POLICY_COEF]

    @property
    def value_coefficient(self):
        return self.hparams[_HP_VALUE_COEF]

    @property
    def ownership_coefficient(self):
        return self.hparams[_HP_OWNERSHIP_COEF]

    @property
    def similarity_coefficient(self):
        return self.hparams[_HP_SIMILARITY_COEF]

    @property
    def label_smoothing(self):
        return self.hparams[_HP_LABEL_SMOOTHING]

    @property
    def discount_factor(self):
        return self.hparams[_HP_DISCOUNT_FACTOR]

    @property
    def learning_rate(self):
        return self.hparams[_HP_LEARNING_RATE]

    @property
    def decay_rate(self):
        return self.hparams[_HP_DECAY_RATE]

    @property
    def weight_decay(self):
        return self.hparams[_HP_WEIGHT_DECAY]

    @property
    def clipnorm(self):
        return self.hparams[_HP_CLIPNORM]

    @property
    def epochs(self):
        return self.hparams[_HP_EPOCHS]

    @property
    def num_early_stopping_warmup_steps(self):
        return self.hparams[_HP_NUM_ES_WARMUP_STEPS]

    @property
    def num_early_stopping_samples(self):
        return self.hparams[_HP_NUM_ES_SAMPLES]

    @property
    def max_early_stopping_slope(self):
        return self.hparams[_HP_MAX_ES_SLOPE]

    def __str__(self):
        return json.dumps({
            'batch_size': self.batch_size,
            'data': self.data,
            'num_unrolls': self.num_unrolls,
            'lz_weights': self.lz_weights,
            'early_stopping': {
                'max_slope': self.max_early_stopping_slope,
                'samples': self.num_early_stopping_samples,
                'num_warmup_steps': self.num_early_stopping_warmup_steps
            },
            'loss': {
                'coefficients': {
                    'ownership': self.ownership_coefficient,
                    'policy': self.policy_coefficient,
                    'similarity': self.similarity_coefficient,
                    'value': self.value_coefficient
                },
                'discount_factor': self.discount_factor,
                'label_smoothing': self.label_smoothing
            },
            'model': {
                'dyn': {
                    'num_blocks': self.num_dyn_blocks,
                    'num_channels': self.num_dyn_channels
                },
                'embeddings_size': self.embeddings_size,
                'repr': {
                    'num_blocks': self.num_repr_blocks,
                    'num_channels': self.num_repr_channels
                }
            },
            'optimizer': {
                'clipnorm': self.clipnorm,
                'decay_rate': self.decay_rate,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay
            }
        }, indent=4, sort_keys=True)
