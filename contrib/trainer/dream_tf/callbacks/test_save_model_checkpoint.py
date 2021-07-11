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

import os
import shutil
import tempfile
import unittest

import tensorflow as tf

from ..test_common import TestUtils
from .save_model_checkpoint import CustomSaveModelCheckpoint

class CustomSaveModelCheckpointTest(unittest.TestCase, TestUtils):
    def setUp(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(1, input_shape=[8]))
        self.model.build([4, 8])
        self.model_dir = tempfile.mkdtemp()
        self.save_model_checkpoint = CustomSaveModelCheckpoint(
            self.model_dir,
            monitor='val_loss'
        )

        self.save_model_checkpoint.set_model(self.model)

    def tearDown(self):
        shutil.rmtree(self.model_dir)

    def test_on_epoch_end(self):
        self.save_model_checkpoint.on_epoch_end(1, { 'val_loss': 0.0 })
        self.assertTrue(os.path.isfile(f'{self.model_dir}/weights.001.h5'))

    def test_save_best_only(self):
        self.save_model_checkpoint.on_epoch_end(1, { 'val_loss': 0.0 })
        self.assertTrue(os.path.isfile(f'{self.model_dir}/weights.001.h5'))

        self.save_model_checkpoint.on_epoch_end(2, { 'val_loss': 1.0 })
        self.assertFalse(os.path.isfile(f'{self.model_dir}/weights.002.h5'))

if __name__ == '__main__':
    unittest.main()
