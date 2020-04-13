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
from .layers.ownership_head import ownership_loss

dream_go_module = tf.load_op_library('libdg_tf.so')

def model_fn(features, labels, mode, params):
    value_hat, policy_hat, next_policy_hat, ownership_hat, tower_hat = tower(features, mode, params)

    if labels:
        # determine the loss for each of the components:
        #
        # - Value head
        # - Policy head (2x)
        # - Ownership
        #
        loss_value = tf.reshape(tf.losses.huber_loss(
            check_numerics(labels['value'], 'value_labels'),
            check_numerics(value_hat, 'value_hat'),
            reduction=tf.losses.Reduction.NONE
        ), (-1, 1))

        loss_policy = tf.reshape(tf.losses.softmax_cross_entropy(
            check_numerics(labels['policy'], 'policy_labels'),
            check_numerics(policy_hat, 'policy_hat'),
            label_smoothing=0.2,
            reduction=tf.losses.Reduction.NONE
        ), (-1, 1))

        loss_next_policy = tf.reshape(tf.losses.softmax_cross_entropy(
            check_numerics(labels['next_policy'], 'next_policy_labels'),
            check_numerics(next_policy_hat, 'next_policy_hat'),
            label_smoothing=0.2,
            reduction=tf.losses.Reduction.NONE
        ), (-1, 1))

        loss_ownership = tf.reshape(ownership_loss(
            labels=check_numerics(labels['ownership'], 'ownership_labels'),
            logits=check_numerics(ownership_hat, 'ownership_hat')
        ), (-1, 1))

        loss_reg = tf.math.accumulate_n(
            inputs=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        )

        loss_unboosted = 0.25 * check_numerics(loss_policy, 'loss_policy') \
                         + 0.25 * check_numerics(loss_next_policy, 'loss_next_policy') \
                         + 1.00 * check_numerics(loss_value, 'loss_value') * labels['boost'] \
                         + 1.00 * check_numerics(loss_ownership, 'loss_ownership')

        loss = tf.reduce_mean(loss_unboosted) + 1e-4 * loss_reg
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

        tf.summary.scalar('loss/policy', tf.reduce_mean(loss_policy))
        tf.summary.scalar('loss/next_policy', tf.reduce_mean(loss_next_policy))
        tf.summary.scalar('loss/value', tf.reduce_mean(loss_value))
        tf.summary.scalar('loss/ownership', tf.reduce_mean(loss_ownership))
        tf.summary.scalar('loss/l2', tf.reduce_mean(loss_reg))

        # image metrics
        def to_heat_image(heat):
            return dream_go_module.tensor_to_heat_image(
                features[0, :, :, 5],
                features[0, :, :, 21],
                heat
            )

        tf.summary.image('value/predictions', to_heat_image(tf.ones([19, 19]) * value_hat[0, 0]))
        tf.summary.image('value/labels', to_heat_image(tf.ones([19, 19]) * labels['value'][0, 0]))
        tf.summary.image('ownership/predictions', to_heat_image(tf.reshape(ownership_hat[0, :], [19, 19])))
        tf.summary.image('ownership/labels', to_heat_image(tf.reshape(labels['ownership'][0, :], [19, 19])))
        tf.summary.image('policy/predictions', to_heat_image(tf.reshape(tf.nn.softmax(policy_hat[0, :361]), [19, 19])))
        tf.summary.image('policy/labels', to_heat_image(tf.reshape(labels['policy'][0, :361], [19, 19])))

        # evaluation metrics such as the accuracy is more human readable than
        # the pure loss function. Even if it is considered bad practice to look
        # at the accuracy instead of the loss.
        policy_hot = tf.argmax(labels['policy'], axis=1)
        next_policy_hot = tf.argmax(labels['next_policy'], axis=1)
        policy_1 = tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, 1), tf.float32)
        policy_3 = tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, 3), tf.float32)
        policy_5 = tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, 5), tf.float32)
        next_policy_1 = tf.cast(tf.nn.in_top_k(next_policy_hat, next_policy_hot, 1), tf.float32)
        next_policy_3 = tf.cast(tf.nn.in_top_k(next_policy_hat, next_policy_hot, 3), tf.float32)
        next_policy_5 = tf.cast(tf.nn.in_top_k(next_policy_hat, next_policy_hot, 5), tf.float32)
        value_1 = tf.cast(tf.equal(tf.sign(value_hat), tf.sign(labels['value'])), tf.float32)
        ownership_1 = tf.cast(tf.equal(tf.sign(ownership_hat), tf.sign(labels['ownership'])), tf.float32)

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
            'accuracy/ownership': tf.metrics.mean(ownership_1),
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
        'value': value_hat,
        'ownership': ownership_hat,
        'policy': tf.nn.softmax(policy_hat),
        'next_policy': tf.nn.softmax(next_policy_hat),
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


def check_numerics(tensor, message, name=None):
    return tf.debugging.check_numerics(tensor, message, name)
