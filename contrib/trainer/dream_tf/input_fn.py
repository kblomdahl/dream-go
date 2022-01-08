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
import tensorflow as tf

from .layers import NUM_FEATURES

dream_go_module = tf.load_op_library('libdg_tf.so')

def _parse(num_unrolls):
    def __do_parse(line):
        lz_features, features, policy, value, ownership, komi, boost, has_ownership = dream_go_module.sgf_to_features(line, num_unrolls)

        labels = {
            'lz_features': lz_features,
            'boost': boost,
            'value': value,
            'policy': policy,
            'ownership': ownership,
            'has_ownership': has_ownership,
            'komi': komi
        }

        return features, labels

    return __do_parse


def _legal_policy(features, labels):
    return tf.greater(tf.reduce_sum(input_tensor=labels['boost']), 0.0)


def _apply_symmetry(symmetry_index, x):
    """ Augment an [n, 19, 19, c] tensor by applying the same transformation to
    each image. """

    def _identity(image):
        return tf.identity(image)

    def _flip_lr(image):
        return tf.reverse(image, [2])

    def _flip_ud(image):
        return tf.reverse(image, [1])

    def _transpose_main(image):
        return tf.transpose(a=image, perm=[0, 2, 1, 3])

    def _transpose_anti(image):
        return tf.reverse(tf.transpose(a=image, perm=[0, 2, 1, 3]), [1, 2])

    def _rot90(image):
        return tf.transpose(a=tf.reverse(image, [2]), perm=[0, 2, 1, 3])

    def _rot180(image):
        return tf.reverse(image, [1, 2])

    def _rot270(image):
        return tf.reverse(tf.transpose(a=image, perm=[0, 2, 1, 3]), [2])

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


def _augment_board(symmetry_index, original_board):
    """ Augment an `[n, 361]` tensor as if it was a `[n, 19, 19, 1]` tensor. """
    return tf.reshape(
        _apply_symmetry(symmetry_index, tf.reshape(original_board, [-1, 19, 19, 1])),
        [-1, 361]
    )


def _augment_policy(symmetry_index, original_policy):
    """ Augment an `[n, 362]` tensor as a board by ignoring the last element. """
    policy, policy_pass = tf.split(original_policy, [361, 1], axis=1)
    policy = _augment_board(symmetry_index, policy)

    return tf.concat([policy, policy_pass], axis=1)


def _augment(features, labels):
    """ Apply a random transformation to the features and each label (where
    relevant) """

    symmetry_index = tf.random.uniform((), 0, 8, tf.int32)
    features = _apply_symmetry(symmetry_index, features)

    labels['lz_features'] = _apply_symmetry(symmetry_index, labels['lz_features'])
    labels['policy'] = _augment_policy(symmetry_index, labels['policy'])
    labels['ownership'] = _augment_board(symmetry_index, labels['ownership'])

    return features, labels


def _fix_history(features, labels):
    """ Zeros out the history planes for 25% of the features. """
    zero_history_mask = np.asarray([1.0] * NUM_FEATURES, 'f2')
    zero_history_mask[3:5] = 0.0
    zero_history_mask = tf.constant(zero_history_mask, tf.float16, [1, 1, 1, NUM_FEATURES])

    random = tf.random.uniform((), 0, 100, tf.int32)
    features = tf.case(
        [
            (tf.less(random, 5), lambda: features * zero_history_mask)
        ],
        default=lambda: features
    )

    return features, labels


def get_dataset(files, *, num_unrolls=1):
    """ Returns a tf.DataSet initializable iterator over the given files """

    with tf.device('cpu:0'):
        num_parallel_calls = tf.data.experimental.AUTOTUNE

        if len(files) > 1:
            file_names = tf.data.Dataset.from_tensor_slices(files)
            dataset = file_names.interleave(
                lambda file: tf.data.TextLineDataset(file),
                cycle_length=16,
                block_length=1,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        else:
            dataset = tf.data.TextLineDataset(files)
        dataset = dataset.map(_parse(num_unrolls), num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(_legal_policy)

        return dataset


def input_fn(files, batch_size, is_training, *, num_unrolls=1, num_test_batches=10):
    dataset = get_dataset(files, num_unrolls=num_unrolls)

    if is_training is True:
        num_parallel_calls = tf.data.experimental.AUTOTUNE

        dataset = dataset.skip(num_test_batches * batch_size)
        dataset = dataset.shuffle(262144 // num_unrolls)
        dataset = dataset.map(_augment, num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(_fix_history, num_parallel_calls=num_parallel_calls)
    elif is_training is False:
        dataset = dataset.take(num_test_batches * batch_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
