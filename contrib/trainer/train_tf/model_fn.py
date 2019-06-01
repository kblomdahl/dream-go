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

import tensorflow as tf
import tensorflow.keras.backend as K

from .optimizers.accum_grad_optimizer import AccumGradOptimizer
from .layers.residual_block import residual_block
from .hooks.dgraph_saver_hook import DGraphSaverHook
from .hooks.increase_global_step import IncreaseGlobalStepHook
from .layers.conv2d_batch_norm import conv2d_batch_norm
from .layers.conv2d_classifier import conv2d_classifier
from .layers.global_pooling_classifier import global_avg_pooling_classifier
from .layers.mb_conv_block import mb_conv_block
from .metrics.slope import avg_slope
from .serializer import serialize_graph


def model_fn(features, labels, mode, params, config):
    with serialize_graph(mode == tf.estimator.ModeKeys.TRAIN):
        x = conv2d_batch_norm(features, params.num_channels, [1, 1], activation='relu')

        for _ in range(params.num_blocks):
            #x = mb_conv_block(x)
            x = residual_block(x)

        predictions = {
            'policy': global_avg_pooling_classifier(x, 362, name='policy'),
            'policy_next': global_avg_pooling_classifier(x, 362, name='policy_next'),
            'value': global_avg_pooling_classifier(x, 2, name='value'),
            'score': global_avg_pooling_classifier(x, 723, name='score'),
            'ownership': conv2d_classifier(x, 2, name='ownership'),
        }

    losses = {
        'policy': tf.keras.losses.categorical_crossentropy(labels['policy'], predictions['policy']),
        'policy_next': tf.keras.losses.categorical_crossentropy(labels['policy_next'], predictions['policy_next']),
        'value': tf.keras.losses.binary_crossentropy(labels['value'], predictions['value']),
        'score': tf.keras.losses.categorical_crossentropy(labels['score'], predictions['score']),
        'ownership': tf.keras.losses.binary_crossentropy(labels['ownership'], predictions['ownership']),
    }

    total_loss = \
          2.72 * K.mean(losses['policy']) \
        + 0.81 * K.mean(losses['policy_next']) \
        + 7.88 * K.mean(losses['value']) \
        + 1.10 * K.mean(losses['score']) \
        + 7.88 * K.mean(losses['ownership'])

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=params.lr)
    #optimizer = AccumGradOptimizer(optimizer, num_iters=params.batch_size // params.mini_batch_size)
    with tf.control_dependencies(update_ops):
        train_ops = optimizer.minimize(
            total_loss,
            global_step=tf.train.get_or_create_global_step()
        )

    # evaluation metrics & TensorBoard summaries
    eval_metric_ops = {}
    for key, prediction in predictions.items():
        eval_metric_ops['accuracy/' + key] = categorical_accuracy(labels[key], prediction, mode)

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops['precision/' + key] = categorical_precision(labels[key], prediction, mode)
            eval_metric_ops['recall/' + key] = categorical_recall(labels[key], prediction, mode)

    for key, value in losses.items():
        mean_value = K.mean(value)

        tf.summary.scalar('loss/' + key, mean_value)
        tf.summary.scalar('slope/loss/' + key, avg_slope(mean_value))

    for key, (_value, update_op) in eval_metric_ops.items():
        tf.summary.scalar(key, update_op)

    training_chief_hooks = [
        #IncreaseGlobalStepHook(),
        DGraphSaverHook(config.model_dir + '/dream_go.json', features, predictions),
    ]

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=tf.group(train_ops),
        eval_metric_ops=eval_metric_ops,
        training_chief_hooks=training_chief_hooks if mode == tf.estimator.ModeKeys.TRAIN else []
    )


def categorical_accuracy(labels, predictions, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        accuracy = K.mean(
            K.cast(
                K.equal(
                    K.argmax(labels, axis=-1),
                    K.argmax(predictions, axis=-1)
                ),
                'float32'
            )
        )

        return accuracy, accuracy
    else:
        return tf.metrics.accuracy(
            K.argmax(labels, axis=-1),
            K.argmax(predictions, axis=-1),
            name='eval/categorical_accuracy'
        )


def categorical_precision(labels, predictions, _mode):
    return tf.metrics.precision(
        K.argmax(labels, axis=-1),
        K.argmax(predictions, axis=-1),
        name='eval/categorical_precision'
    )


def categorical_recall(labels, predictions, _mode):
    return tf.metrics.recall(
        K.argmax(labels, axis=-1),
        K.argmax(predictions, axis=-1),
        name='eval/categorical_recall'
    )
