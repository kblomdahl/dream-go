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

import io
import json
import unittest

from .model_config import ModelConfig

MODEL_CONFIG_JSON = '''
{
    "batch_size": 8196,
    "num_unrolls": 1,
    "lz_weights": [ "a", "b" ],
    "data": [ "c", "d" ],

    "model": {
        "embeddings_size": 32,
        "representation": {
            "num_blocks": 12,
            "num_channels": 24
        },
        "transition_predictor": {
            "layers": 6
        },
        "predictor": {
            "layers": 8
        }
    },

    "loss": {
        "coefficients": {
            "policy": 0.5,
            "value": 0.6,
            "target": 0.4,
            "similarity": 0.2
        },
        "label_smoothing": 1.0,
        "discount_factor": 0.7
    },

    "optimizer": {
        "learning_rate": 0.1,
        "decay_rate": 0.8,
        "weight_decay": 1e-2,
        "clipnorm": 1.0,
        "epochs": 5
    },

    "early_stopping": {
        "num_warmup_steps": 10,
        "samples": 2,
        "max_slope": -1e-8
    }
}
'''

class ModelConfigTest(unittest.TestCase):
    def setUp(self):
        self.filename = io.StringIO(MODEL_CONFIG_JSON)

    def tearDown(self):
        pass

    def test_str(self):
        s = str(ModelConfig(self.filename))
        self.maxDiff = 10000
        self.assertEqual(json.loads(s), json.loads(MODEL_CONFIG_JSON))

    def test_batch_size(self):
        self.assertEqual(ModelConfig(self.filename).batch_size, 8196)

    def test_num_unrolls(self):
        self.assertEqual(ModelConfig(self.filename).num_unrolls, 1)

    def test_lz_weights(self):
        self.assertEqual(ModelConfig(self.filename).lz_weights, [ "a", "b" ])

    def test_data(self):
        self.assertEqual(ModelConfig(self.filename).data, [ "c", "d" ])

    def test_embeddings_size(self):
        self.assertEqual(ModelConfig(self.filename).embeddings_size, 32)

    def test_num_repr_blocks(self):
        self.assertEqual(ModelConfig(self.filename).num_repr_blocks, 12)

    def test_num_repr_channels(self):
        self.assertEqual(ModelConfig(self.filename).num_repr_channels, 24)

    def test_num_trans_layers(self):
        self.assertEqual(ModelConfig(self.filename).num_trans_layers, 6)

    def test_num_pred_layers(self):
        self.assertEqual(ModelConfig(self.filename).num_pred_layers, 8)

    def test_policy_coefficient(self):
        self.assertEqual(ModelConfig(self.filename).policy_coefficient, 0.5)

    def test_value_coefficient(self):
        self.assertEqual(ModelConfig(self.filename).value_coefficient, 0.6)

    def test_target_coefficient(self):
        self.assertEqual(ModelConfig(self.filename).target_coefficient, 0.4)

    def test_similarity_coefficient(self):
        self.assertEqual(ModelConfig(self.filename).similarity_coefficient, 0.2)

    def test_label_smoothing(self):
        self.assertEqual(ModelConfig(self.filename).label_smoothing, 1.0)

    def test_discount_factor(self):
        self.assertEqual(ModelConfig(self.filename).discount_factor, 0.7)

    def test_learning_rate(self):
        self.assertEqual(ModelConfig(self.filename).learning_rate, 0.1)

    def test_decay_rate(self):
        self.assertEqual(ModelConfig(self.filename).decay_rate, 0.8)

    def test_weight_decay(self):
        self.assertEqual(ModelConfig(self.filename).weight_decay, 1e-2)

    def test_clipnorm(self):
        self.assertEqual(ModelConfig(self.filename).clipnorm, 1.0)

    def test_epochs(self):
        self.assertEqual(ModelConfig(self.filename).epochs, 5)

    def test_num_early_stopping_warmup_steps(self):
        self.assertEqual(ModelConfig(self.filename).num_early_stopping_warmup_steps, 10)

    def test_num_early_stopping_samples(self):
        self.assertEqual(ModelConfig(self.filename).num_early_stopping_samples, 2)

    def test_max_early_stopping_slope(self):
        self.assertEqual(ModelConfig(self.filename).max_early_stopping_slope, -1e-8)
