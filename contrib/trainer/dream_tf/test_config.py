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

import unittest

from .config import Config
from .test_common import TestUtils

class ConfigTest(unittest.TestCase, TestUtils):
    def test_is_start(self):
        self.assertEqual(Config(['--start'], exit_on_error=False).is_start(), True)

    def test_is_resume(self):
        self.assertEqual(Config(['--resume'], exit_on_error=False).is_resume(), True)

    def test_is_verify(self):
        self.assertEqual(Config(['--verify'], exit_on_error=False).is_verify(), True)

    def test_is_dump(self):
        self.assertEqual(Config(['--dump'], exit_on_error=False).is_dump(), True)

    def test_files(self):
        self.assertEqual(Config(['--start', 'x', 'y'], exit_on_error=False).files, ['x', 'y'])

    def test_model(self):
        self.assertEqual(Config(['--start', '--model', 'x'], exit_on_error=False).get_model_dir(), 'x')

    def test_model_default(self):
        self.assertIsNotNone(Config(['--start'], exit_on_error=False).get_model_dir())

    def test_name(self):
        self.assertIn('primex', Config(['--start', '--name', 'primex'], exit_on_error=False).get_model_dir())

    def test_warm_start(self):
        self.assertEqual(Config(['--start', '--warm-start', 'x'], exit_on_error=False).warm_start, 'x')

    def test_lz_weights(self):
        self.assertEqual(Config(['--start', '--lz-weights', 'x'], exit_on_error=False).lz_weights, 'x')

    def test_batch_size(self):
        self.assertEqual(Config(['--start', '--batch-size', '32'], exit_on_error=False).batch_size, 32)

    def test_num_unrolls(self):
        self.assertEqual(Config(['--start', '--num-unrolls', '2'], exit_on_error=False).num_unrolls, 2)

    def test_num_channels(self):
        self.assertEqual(Config(['--start', '--num-channels', '8'], exit_on_error=False).num_channels, 8)

    def test_num_dynamics_channels(self):
        self.assertEqual(Config(['--start', '--num-dynamics-channels', '8'], exit_on_error=False).num_dynamics_channels, 8)

    def test_num_blocks(self):
        self.assertEqual(Config(['--start', '--num-blocks', '6'], exit_on_error=False).num_blocks, 6)

    def test_num_dynamics_blocks(self):
        self.assertEqual(Config(['--start', '--num-dynamics-blocks', '2'], exit_on_error=False).num_dynamics_blocks, 2)

    def test_embeddings_size(self):
        self.assertEqual(Config(['--start', '--embeddings-size', '255'], exit_on_error=False).embeddings_size, 255)

    def test_policy_coefficient(self):
        self.assertEqual(Config(['--start', '--policy-coefficient', '0.2'], exit_on_error=False).policy_coefficient, 0.2)

    def test_value_coefficient(self):
        self.assertEqual(Config(['--start', '--value-coefficient', '0.2'], exit_on_error=False).value_coefficient, 0.2)

    def test_ownership_coefficient(self):
        self.assertEqual(Config(['--start', '--ownership-coefficient', '0.2'], exit_on_error=False).ownership_coefficient, 0.2)

    def test_similarity_coefficient(self):
        self.assertEqual(Config(['--start', '--similarity-coefficient', '0.2'], exit_on_error=False).similarity_coefficient, 0.2)

    def test_discount_factor(self):
        self.assertEqual(Config(['--start', '--discount-factor', '0.5'], exit_on_error=False).discount_factor, 0.5)

    def test_weight_decay(self):
        self.assertEqual(Config(['--start', '--weight-decay', '0.01'], exit_on_error=False).weight_decay, 0.01)

    def test_label_smoothing(self):
        self.assertEqual(Config(['--start', '--label-smoothing', '0.0'], exit_on_error=False).label_smoothing, 0.0)

    def test_initial_learning_rate(self):
        self.assertEqual(Config(['--start', '--initial-learning-rate', '0.1'], exit_on_error=False).initial_learning_rate, 0.1)

    def test_num_es_warmup_steps(self):
        self.assertEqual(Config(['--start', '--num-es-warmup-steps', '2000'], exit_on_error=False).num_es_warmup_steps, 2000)

    def test_num_es_samples(self):
        self.assertEqual(Config(['--start', '--num-es-samples', '5'], exit_on_error=False).num_es_samples, 5)

    def test_max_es_slope(self):
        self.assertEqual(Config(['--start', '--max-es-slope', '-0.0001'], exit_on_error=False).max_es_slope, -0.0001)

    def test_clipnorm(self):
        self.assertEqual(Config(['--start', '--clipnorm', '0.1'], exit_on_error=False).clipnorm, 0.1)

    def test_epochs(self):
        self.assertEqual(Config(['--start', '--epochs', '100'], exit_on_error=False).epochs, 100)

    def test_run_eagerly(self):
        self.assertEqual(Config(['--start', '--run-eagerly'], exit_on_error=False).run_eagerly, True)

    def test_hparams(self):
        self.assertIsNotNone(Config(['--start']).hparams)

if __name__ == '__main__':
    unittest.main()
