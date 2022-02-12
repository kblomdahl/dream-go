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

import io
import os.path
import tempfile
import unittest

import tensorflow as tf

from .layers import NUM_FEATURES
from .model import DreamGoNet
from .test_common import TestUtils

class DreamGoNetBase(TestUtils):
    @property
    def inputs(self):
        return tf.repeat(tf.random.uniform([1, self.num_unrolls, 19, 19, NUM_FEATURES], dtype=tf.float16), self.batch_size, axis=0)

    @property
    def labels(self):
        return {
            'lz_features': tf.repeat(tf.random.uniform([1, self.num_unrolls, 19, 19, 18], dtype=tf.float16), self.batch_size, axis=0),
            'boost': tf.repeat(tf.random.uniform([1, self.num_unrolls, 1], dtype=tf.float32), self.batch_size, axis=0),
            'value': tf.repeat(tf.random.uniform([1, self.num_unrolls, 1], dtype=tf.float32), self.batch_size, axis=0),
            'policy': tf.repeat(tf.random.uniform([1, self.num_unrolls, 362], dtype=tf.float32), self.batch_size, axis=0),
            'ownership': tf.repeat(tf.random.uniform([1, self.num_unrolls, 361], dtype=tf.float32), self.batch_size, axis=0),
            'has_ownership': tf.repeat(tf.round(tf.random.uniform([1, self.num_unrolls, 1], dtype=tf.float32)), self.batch_size, axis=0),
            'komi': tf.repeat(tf.random.uniform([1, self.num_unrolls, 1], dtype=tf.float32), self.batch_size, axis=0)
        }

    def test_using_cudnn_rnn(self):
        self.assertTrue(self.model.rnn._could_use_gpu_kernel)

    def test_dtype(self):
        y = self.model(self.x)
        self.assertEqual(y['value'].dtype, tf.float32)
        self.assertEqual(y['policy'].dtype, tf.float32)
        self.assertEqual(y['ownership'].dtype, tf.float32)
        self.assertEqual(y['tower'].dtype, tf.float16)

    def test_shape(self):
        y = self.model(self.x)
        self.assertEqual(y['value'].shape, [self.batch_size * self.num_unrolls, 1])
        self.assertEqual(y['policy'].shape, [self.batch_size * self.num_unrolls, 362])
        self.assertEqual(y['ownership'].shape, [self.batch_size * self.num_unrolls, 361])
        self.assertEqual(y['tower'].shape, [self.batch_size * self.num_unrolls, self.embeddings_size])

    def test_fit(self):
        losses = self.fit_model(
            self.model,
            inputs=self.inputs,
            labels=self.labels
        )

        self.assertDecreasing(losses)

    def test_evaluate(self):
        metrics = self.model.evaluate(
            tf.data.Dataset.from_tensors((self.inputs, self.labels)),
            return_dict=True,
            verbose=0
        )

        self.assertIn('loss', metrics)
        self.assertIn('loss/value', metrics)
        self.assertIn('loss/policy', metrics)
        self.assertIn('loss/ownership', metrics)
        self.assertIn('loss/l2', metrics)

        for i in range(self.num_unrolls):
            self.assertIn(f'value/accuracy/[{i}]', metrics)
            self.assertIn(f'policy/accuracy/[{i}]', metrics)
            self.assertIn(f'ownership/accuracy/[{i}]', metrics)

    def test_as_dict(self):
        out = io.StringIO('')
        self.model(self.inputs)
        self.model.dump_to(out)

        #self.assertIn('num_channels:0', out.getvalue())
        #self.assertIn('num_samples:0', out.getvalue())
        #self.assertIn('01_upsample/conv_1:0', out.getvalue())
        #self.assertIn('01_upsample/conv_1/offset:0', out.getvalue())
        #self.assertIn('05v_value/conv_1:0', out.getvalue())
        #self.assertIn('05v_value/conv_1/offset:0', out.getvalue())
        #self.assertIn('05v_value/linear_2:0', out.getvalue())
        #self.assertIn('05v_value/linear_2/offset:0', out.getvalue())
        #self.assertIn('05p_policy/conv_1:0', out.getvalue())
        #self.assertIn('05p_policy/conv_1/offset:0', out.getvalue())
        #self.assertIn('05p_policy/linear_1:0', out.getvalue())
        #self.assertIn('05p_policy/linear_1/offset:0', out.getvalue())

class DreamGoNetTest(unittest.TestCase, DreamGoNetBase):
    def setUp(self):
        self.batch_size = 5
        self.num_channels = 48
        self.embeddings_size = 32
        self.num_unrolls = 3
        self.model = DreamGoNet(
            batch_size=self.batch_size,
            num_repr_blocks=2,
            num_repr_channels=self.num_channels,
            num_dyn_blocks=2,
            num_dyn_channels=self.num_channels,
            embeddings_size=self.embeddings_size,
            num_unrolls=self.num_unrolls,
            label_smoothing=0.0
        )
        self.x = tf.zeros([self.batch_size, self.num_unrolls, 19, 19, NUM_FEATURES], tf.float16)

    def test_save_weights(self):
        self.model(self.x, training=False)  # build the layers
        with tempfile.TemporaryDirectory() as dir_name:
            self.model.save_weights(f'{dir_name}/weights.001')
            self.assertTrue(os.path.isfile(f'{dir_name}/weights.001.index'))

    def test_metrics(self):
        inputs = self.inputs
        outputs = self.model(inputs, training=False)
        labels = {
            'value': tf.reshape(outputs['value'], self.labels['value'].shape),
            'policy': tf.reshape(tf.nn.softmax(outputs['policy']), self.labels['policy'].shape),
            'ownership': tf.reshape(outputs['ownership'], self.labels['ownership'].shape),
            'has_ownership': tf.ones(self.labels['has_ownership'].shape, tf.float32)
        }

        metrics = self.model.evaluate(
            tf.data.Dataset.from_tensors((inputs, labels)),
            return_dict=True,
            verbose=0
        )

        self.assertAlmostEqual(metrics['loss/value'], 0.0, delta=1e-4)
        #self.assertAlmostEqual(metrics['loss/policy'], 5.85, delta=0.25)  # these seem noisy
        #self.assertAlmostEqual(metrics['loss/ownership'], 0.6855, delta=0.1)  # these seem noisy
        self.assertGreater(metrics['loss/l2'], 0.0)

        self.assertAlmostEqual(metrics['value/accuracy/[0]'], 1.0)
        self.assertAlmostEqual(metrics['policy/accuracy/[0]'], 1.0)
        self.assertAlmostEqual(metrics['ownership/accuracy/[0]'], 1.0, delta=0.05)

class DreamGoNetLzTest(unittest.TestCase, DreamGoNetBase):
    def setUp(self):
        self.batch_size = 5
        self.num_channels = 48
        self.embeddings_size = 32
        self.num_unrolls = 3
        self.model = DreamGoNet(
            batch_size=self.batch_size,
            num_repr_blocks=2,
            num_repr_channels=self.num_channels,
            num_dyn_blocks=2,
            num_dyn_channels=self.num_channels,
            embeddings_size=self.embeddings_size,
            num_unrolls=self.num_unrolls,
            label_smoothing=0.0,
            lz_weights='fixtures/d645af9.gz'
        )
        self.x = tf.zeros([self.batch_size, self.num_unrolls, 19, 19, NUM_FEATURES], tf.float16)

if __name__ == '__main__':
    unittest.main()
