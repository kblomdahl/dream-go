# Copyright (c) 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from .layers.conv2d_batch_norm import conv2d_batch_norm
from .layers.conv2d_classifier import conv2d_classifier
from .layers.global_pooling_classifier import global_avg_pooling_classifier
from .layers.residual_block import residual_block
from .optimizer import AccumulatedNadam


def model_fn(opts, inputs_shape):
    # output heads
    inputs = Input(shape=inputs_shape[1:])
    x = conv2d_batch_norm(inputs, opts.num_channels, [1, 1], activation='relu')

    for _ in range(opts.num_blocks):
        x = residual_block(x)

    model = Model(
        inputs=inputs,
        outputs=[
            global_avg_pooling_classifier(x, 362, name='policy'),
            global_avg_pooling_classifier(x, 362, name='policy_next'),
            global_avg_pooling_classifier(x, 2, name='value'),
            global_avg_pooling_classifier(x, 722, name='score'),
            conv2d_classifier(x, 2, name='ownership')
        ]
    )

    # losses and accuracy
    model.compile(
        optimizer=AccumulatedNadam(
            update_freq=opts.batch_size // opts.mini_batch_size,
            learning_rate=opts.lr
        ),
        loss={
            'policy': 'categorical_crossentropy',
            'policy_next': 'categorical_crossentropy',
            'value': 'binary_crossentropy',
            'score': 'categorical_crossentropy',
            'ownership': 'binary_crossentropy',
        },
        loss_weights={
            'policy': 2.72,  # 70%
            'policy_next': 0.81,  # 30%
            'value': 7.88,  # 90%
            'score': 1.10,  # 40%
            'ownership': 7.88,  # 90%
        },
        metrics={
            'policy': 'categorical_accuracy',
            'policy_next': 'categorical_accuracy',
            'value': 'binary_accuracy',
            'score': 'categorical_accuracy',
            'ownership': 'binary_accuracy',
        },
        run_eagerly=opts.profile,
    )

    return model