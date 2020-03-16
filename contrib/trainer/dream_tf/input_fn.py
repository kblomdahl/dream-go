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

import numpy as np
import os
import tensorflow as tf

from .constants.boost_table import BOOST_PER_MOVE_NUMBER
from .ffi.libdg_go import get_single_example, set_seed
from .layers import NUM_FEATURES


def _parse(is_deterministic):
    def __parse(raw_example):
        result, example = get_single_example(raw_example)

        if result != 0:
            features = np.zeros((19, 19, NUM_FEATURES), 'f2')

            boost = np.zeros((), 'f4')
            value = np.zeros((), 'f4')
            policy = np.zeros((362,), 'f4')
            next_policy = np.zeros((362,), 'f4')
            ownership = np.zeros((361,), 'f4')
            komi = np.zeros((), 'f4')
        else:
            features = np.frombuffer(example['features'], 'f2').copy()

            value = np.asarray(1.0 if example['color'] == example['winner'] else -1.0, 'f4')
            policy = np.frombuffer(example['policy'], 'f4').copy()
            next_policy = np.frombuffer(example['next_policy'], 'f4').copy()
            ownership = np.frombuffer(example['ownership'], 'f4').copy()
            komi = np.asarray(example['komi'], 'f4')
            if example['color'] == 1:  # is black
                komi = -komi

            if example['number'] <= len(BOOST_PER_MOVE_NUMBER):
                boost = np.asarray(BOOST_PER_MOVE_NUMBER[example['number'] - 1], 'f4')
            else:
                boost = np.asarray(1.0, 'f4')

            # fix any partial policy
            policy[example['index']] += 1.0 - np.sum(policy)
            next_policy[example['next_index']] += 1.0 - np.sum(next_policy)

        return features, boost, value, policy, next_policy, ownership, komi

    def __do_parse(line):
        features, boost, value, policy, next_policy, ownership, komi = tuple(tf.py_func(
            __parse,
            [line],
            [tf.float16, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        ))

        features = tf.reshape(features, [19, 19, NUM_FEATURES])
        labels = {
            'boost': tf.reshape(boost, [1]),
            'value': tf.reshape(value, [1]),
            'policy': tf.reshape(policy, [362]),
            'next_policy': tf.reshape(next_policy, [362]),
            'ownership': tf.reshape(ownership, [361]),
            'komi': tf.reshape(komi, [1])
        }

        return features, labels

    # by default the random number generator is seeded from entropy, but if we
    # are running deterministically. Seed it manually instead*.
    #
    # * Since the Tensorflow graph seed has not been set yet :'(
    if is_deterministic:
        set_seed(0x454f1317)

    return __do_parse


def _illegal_policy(features, labels):
    return tf.greater(tf.reduce_sum(labels['boost']), 0.0)


def _apply_symmetry(symmetry_index, x):
    def _identity(image):
        return tf.identity(image)

    def _flip_lr(image):
        return tf.reverse_v2(image, [1])

    def _flip_ud(image):
        return tf.reverse_v2(image, [0])

    def _transpose_main(image):
        return tf.transpose(image, [1, 0, 2])

    def _transpose_anti(image):
        return tf.reverse_v2(tf.transpose(image, [1, 0, 2]), [0, 1])

    def _rot90(image):
        return tf.transpose(tf.reverse_v2(image, [1]), [1, 0, 2])

    def _rot180(image):
        return tf.reverse_v2(image, [0, 1])

    def _rot270(image):
        return tf.reverse_v2(tf.transpose(image, [1, 0, 2]), [1])

    return tf.case(
        [
            (tf.equal(symmetry_index, 0), lambda: _identity(x)),
            (tf.equal(symmetry_index, 1), lambda: _flip_lr(x)),
            (tf.equal(symmetry_index, 2), lambda: _flip_ud(x)),
            (tf.equal(symmetry_index, 3), lambda: _transpose_main(x)),
            (tf.equal(symmetry_index, 4), lambda: _transpose_anti(x)),
            (tf.equal(symmetry_index, 5), lambda: _rot90(x)),
            (tf.equal(symmetry_index, 6), lambda: _rot180(x)),
            (tf.equal(symmetry_index, 7), lambda: _rot270(x)),
        ],
        None,
        exclusive=True
    )


def _augment(features, labels):
    # apply a random transformation to the input features
    symmetry_index = tf.random_uniform((), 0, 8, tf.int32)
    features = _apply_symmetry(symmetry_index, features)

    # transforming the policy is _harder_ since it has that extra pass
    # element at the end, so we temporarily remove it while the tensor gets
    # a random transformation applied
    policy, policy_pass = tf.split(labels['policy'], (361, 1))
    policy = tf.reshape(_apply_symmetry(symmetry_index, tf.reshape(policy, [19, 19, 1])), [361])
    next_policy, next_policy_pass = tf.split(labels['next_policy'], (361, 1))
    next_policy = tf.reshape(_apply_symmetry(symmetry_index, tf.reshape(next_policy, [19, 19, 1])), [361])
    ownership = tf.reshape(_apply_symmetry(symmetry_index, tf.reshape(labels['ownership'], [19, 19, 1])), [361])

    labels['policy'] = tf.concat([policy, policy_pass], 0)
    labels['next_policy'] = tf.concat([next_policy, next_policy_pass], 0)
    labels['ownership'] = tf.nn.softmax(0.5 + 0.5 * ownership)

    return features, labels


def _fix_history(features, labels):
    """ Zeros out the history planes for 25% of the features. """
    zero_history_mask = np.asarray([1.0] * NUM_FEATURES, 'f2')
    zero_history_mask[3:5] = 0.0
    zero_history_mask = tf.constant(zero_history_mask, tf.float16, (1, 1, NUM_FEATURES))

    random = tf.random_uniform((), 0, 100, tf.int32)
    features = tf.case(
        [
            (tf.less(random, 5), lambda: features * zero_history_mask)
        ],
        default=lambda: features
    )

    return features, labels


def get_dataset(files, is_training=True, is_deterministic=False):
    """ Returns a tf.DataSet initializable iterator over the given files """

    with tf.device('cpu:0'):
        num_parallel_calls = max(os.cpu_count() - 8, 4) if (is_training and not is_deterministic) else 1

        if len(files) > 1:
            file_names = tf.data.Dataset.from_tensor_slices(files)
            dataset = file_names.interleave(
                lambda file: tf.data.TextLineDataset(file),
                cycle_length=16,
                block_length=1,
                num_parallel_calls=16
            )
        else:
            dataset = tf.data.TextLineDataset(files)
        dataset = dataset.map(_parse(is_deterministic), num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(_illegal_policy)
        if is_training:
            dataset = dataset.shuffle(262144)
            dataset = dataset.repeat()
            dataset = dataset.map(_augment, num_parallel_calls=4)
            dataset = dataset.map(_fix_history, num_parallel_calls=4)

        return dataset


def input_fn(files, batch_size, features_mask, is_training, is_deterministic=False):
    dataset = get_dataset(files, is_training, is_deterministic)

    if features_mask is not None:
        features_mask = tf.constant(features_mask, tf.float16, (1, 1, NUM_FEATURES))

        def _mask_features(features, labels):
            return features * features_mask, labels

        dataset = dataset.map(_mask_features)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)

    return dataset
