# Copyright (c) 2020 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

class TestUtils(object):
    def tearDown(self):
        tf.keras.backend.clear_session()

    def create_categorical_labels(self, shape):
        cand = np.random.random(shape)
        return cand / np.sum(cand)

    def fit_categorical(self, *, inputs, outputs, labels, learning_rate=None, min_steps=None, max_steps=None):
        return self.fit(
            inputs=inputs,
            outputs=outputs,
            labels=labels,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            learning_rate=learning_rate,
            min_steps=min_steps,
            max_steps=max_steps
        )

    def fit_regression(self, *, inputs, outputs, labels, learning_rate=None, min_steps=None, max_steps=None):
        return self.fit(
            inputs=inputs,
            outputs=outputs,
            labels=labels,
            loss=tf.keras.losses.MeanSquaredError(),
            learning_rate=learning_rate,
            min_steps=min_steps,
            max_steps=max_steps
        )

    def fit(self, *, inputs, outputs, labels, loss, learning_rate, min_steps=None, max_steps=None):
        input_layer = tf.keras.Input(shape=inputs.shape[1:])
        model = tf.keras.Model(
            inputs=input_layer,
            outputs=outputs(input_layer)
        )
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate or 1e-4),
            loss=loss
        )

        return self.fit_model(
            model,
            inputs=inputs,
            labels=labels,
            min_steps=min_steps,
            max_steps=max_steps
        )

    def fit_model(self, model, *, inputs, labels, min_steps=None, max_steps=None):
        early_stopping = EarlyLossStopping(min_batches=min_steps or 5)
        model.fit(
            tf.data.Dataset.from_tensors((inputs, labels)).repeat(max_steps or 25),
            callbacks=[early_stopping],
            verbose=0
        )

        return early_stopping.losses

    def assertDecreasing(self, losses, threshold=0):
        self.assertLess(get_slope(losses), threshold)

def get_slope(losses):
    """ Returns the slope of a 1-degree polynomial fit to the given points """
    return np.polyfit(
        np.arange(len(losses)),
        losses,
        1
    )[0]

class EarlyLossStopping(tf.keras.callbacks.Callback):
    def __init__(self, min_batches=5, threshold=0):
        super(EarlyLossStopping, self).__init__()

        self.losses = []
        self.min_batches = min_batches
        self.threshold = threshold

    def on_train_batch_end(self, batch, logs):
        self.losses.append(logs['loss'])

        if len(self.losses) > self.min_batches and get_slope(self.losses) < self.threshold:
            self.model.stop_training = True
