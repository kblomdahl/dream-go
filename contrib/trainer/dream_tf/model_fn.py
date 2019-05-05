# Copyright (c) 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

from .hooks.dump import DUMP_OPS
from .hooks.learning_rate import LEARNING_RATE, LOSS
from .layers.tower import tower

def reduce_head(heads, f):
    alpha = 1.0
    results = []

    for h in reversed(heads):
        results.append(alpha * f(h))
        alpha /= 1.5

    return tf.reduce_sum(results, axis=0)

def model_fn(features, labels, mode, params):
    value_hat, ownership_hat, policy_hat, next_policy_hat, tower_hat = tower(features, mode, params)

    if labels:
        # determine the loss for each of the components:
        #
        # - Value head
        # - Ownership head
        # - Policy head (2x)
        #
        loss_value = reduce_head(value_hat, lambda v: tf.reshape(tf.squared_difference(
            tf.check_numerics(labels['value'], 'value_labels'),
            tf.check_numerics(v, 'value_hat')
        ), (-1, 1)))

        loss_ownership = reduce_head(ownership_hat, lambda o: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.nn.softmax(0.5 + 0.5 * tf.check_numerics(tf.stack([labels['ownership'], -labels['ownership']], axis=2), 'ownership_labels')),
            logits=tf.check_numerics(0.5 + 0.5 * tf.stack([o, -o], axis=2), 'ownership_hat')
        ), axis=1, keep_dims=True))

        loss_policy = reduce_head(policy_hat, lambda p: tf.reshape(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.check_numerics(labels['policy'], 'policy_labels'),
            logits=tf.check_numerics(p, 'policy_hat')
        ), (-1, 1)))

        loss_next_policy = reduce_head(next_policy_hat, lambda pn: tf.reshape(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.check_numerics(labels['next_policy'], 'next_policy_labels'),
            logits=tf.check_numerics(pn, 'next_policy_hat')
        ), (-1, 1)))

        loss_unboosted = 1.00 * loss_policy \
                         + 0.25 * loss_next_policy \
                         + 1.50 * loss_value \
                         + 1.00 * loss_ownership

        loss = tf.reduce_mean(labels['boost'] * loss_unboosted)
        tf.add_to_collection(LOSS, loss_unboosted)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # set an initial learning rate and then rely on the `LearningRateScheduler`
            # hook to decrease it when the loss plateaus
            learning_rate = tf.Variable(params['learning_rate'], False, name='lr')
            loss_scale = 128

            tf.add_to_collection(LEARNING_RATE, learning_rate)

            global_step = tf.train.get_global_step()
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                variables = tf.trainable_variables()
                gradients = tf.gradients(
                    loss_scale * loss,
                    variables,
                    gate_gradients=True,
                    aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
                    #colocate_gradients_with_ops=True  # would force _custom gradients_ to the CPU
                )

                gradients = [grad / loss_scale for grad in gradients]
                train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)

            # during training it is very useful to plot the norm of the gradients at
            # each tensor so that we can detect the cause of any exploding gradients
            # or similar issues.
            for grad, var in zip(gradients, variables):
                var_name = var.op.name

                if grad is not None:
                    tf.summary.scalar('gradients/' + var_name, tf.norm(grad))
                tf.summary.scalar('norms/' + var_name, tf.norm(var))

            tf.summary.scalar('learning_rate', learning_rate)
        else:
            train_op = None

        tf.summary.histogram('activations/policy', policy_hat)
        tf.summary.histogram('activations/next_policy', next_policy_hat)
        tf.summary.histogram('activations/value', value_hat)
        tf.summary.histogram('activations/ownership', ownership_hat)

        tf.summary.scalar('loss/policy', tf.reduce_mean(loss_policy))
        tf.summary.scalar('loss/next_policy', tf.reduce_mean(loss_next_policy))
        tf.summary.scalar('loss/value', tf.reduce_mean(loss_value))
        tf.summary.scalar('loss/ownership', tf.reduce_mean(loss_ownership))

        # evaluation metrics such as the accuracy is more human readable than
        # the pure loss function. Even if it is considered bad practice to look
        # at the accuracy instead of the loss.
        policy_hot = tf.argmax(labels['policy'], axis=1)
        next_policy_hot = tf.argmax(labels['next_policy'], axis=1)
        policy_1 = tf.cast(tf.nn.in_top_k(policy_hat[-1], policy_hot, 1), tf.float32)
        policy_3 = tf.cast(tf.nn.in_top_k(policy_hat[-1], policy_hot, 3), tf.float32)
        policy_5 = tf.cast(tf.nn.in_top_k(policy_hat[-1], policy_hot, 5), tf.float32)
        next_policy_1 = tf.cast(tf.nn.in_top_k(next_policy_hat[-1], next_policy_hot, 1), tf.float32)
        next_policy_3 = tf.cast(tf.nn.in_top_k(next_policy_hat[-1], next_policy_hot, 3), tf.float32)
        next_policy_5 = tf.cast(tf.nn.in_top_k(next_policy_hat[-1], next_policy_hot, 5), tf.float32)
        value_1 = tf.cast(tf.equal(tf.sign(labels['value']), tf.sign(value_hat[-1])), tf.float32)
        ownership_1 = tf.cast(tf.equal(tf.sign(labels['ownership']), tf.sign(ownership_hat[-1])), tf.float32)

        tf.summary.scalar('accuracy/policy_1', tf.reduce_mean(policy_1))
        tf.summary.scalar('accuracy/policy_3', tf.reduce_mean(policy_3))
        tf.summary.scalar('accuracy/policy_5', tf.reduce_mean(policy_5))
        tf.summary.scalar('accuracy/next_policy_1', tf.reduce_mean(next_policy_1))
        tf.summary.scalar('accuracy/next_policy_3', tf.reduce_mean(next_policy_3))
        tf.summary.scalar('accuracy/next_policy_5', tf.reduce_mean(next_policy_5))
        tf.summary.scalar('accuracy/value', tf.reduce_mean(value_1))
        tf.summary.scalar('accuracy/ownership', tf.reduce_mean(ownership_1))

        eval_metric_ops = {
            'accuracy/policy_1': tf.metrics.mean(policy_1),
            'accuracy/policy_3': tf.metrics.mean(policy_3),
            'accuracy/policy_5': tf.metrics.mean(policy_5),
            'accuracy/next_policy_1': tf.metrics.mean(next_policy_1),
            'accuracy/next_policy_3': tf.metrics.mean(next_policy_3),
            'accuracy/next_policy_5': tf.metrics.mean(next_policy_5),
            'accuracy/value': tf.metrics.mean(value_1),
            'loss/policy': tf.metrics.mean(loss_policy),
            'loss/next_policy': tf.metrics.mean(loss_next_policy),
            'loss/value': tf.metrics.mean(loss_value),
            'loss/ownership': tf.metrics.mean(loss_ownership)
        }
    else:
        loss = None
        train_op = None
        eval_metric_ops = {}

    # output the predictions, and some other intermediate tensors that may
    # be useful.
    predictions = {
        'features': features,
        'value': value_hat[-1],
        'policy': tf.nn.softmax(policy_hat[-1]),
        'next_policy': tf.nn.softmax(next_policy_hat[-1]),
        'ownership': ownership_hat[-1],
        'tower': tower_hat
    }

    # get ride of _worthless_ collections that would just clutter up the
    # saved graph. We do this here to avoid sprinkling a lot of conditions all
    # over the code.
    if mode != tf.estimator.ModeKeys.PREDICT:
        tf.get_default_graph().clear_collection(DUMP_OPS)

    # put it all together into a specification
    return tf.estimator.EstimatorSpec(
        mode,
        predictions,
        loss,
        train_op,
        eval_metric_ops
    )
