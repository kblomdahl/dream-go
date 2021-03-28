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

from .ffi.libdg_go import set_seed
from .layers import NUM_FEATURES

dream_go_module = tf.load_op_library('libdg_tf.so')

def _parse(is_deterministic):
    def __do_parse(line):
        lz_features, features, policy, next_policy, value, ownership, komi, boost, has_ownership = dream_go_module.sgf_to_features(line)

        labels = {
            'lz_features': lz_features,
            'boost': tf.reshape(boost, [1]),
            'value': tf.reshape(value, [1]),
            'policy': tf.reshape(policy, [362]),
            'next_policy': tf.reshape(next_policy, [362]),
            'ownership': tf.reshape(ownership, [361]),
            'has_ownership': tf.reshape(has_ownership, [1]),
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


def _legal_policy(features, labels):
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
    lz_features = _apply_symmetry(symmetry_index, labels['lz_features'])

    # transforming the policy is _harder_ since it has an extra pass
    # element at the end, so we temporarily remove it while the tensor gets
    # a random transformation applied
    policy, policy_pass = tf.split(labels['policy'], (361, 1))
    policy = tf.reshape(_apply_symmetry(symmetry_index, tf.reshape(policy, [19, 19, 1])), [361])
    next_policy, next_policy_pass = tf.split(labels['next_policy'], (361, 1))
    next_policy = tf.reshape(_apply_symmetry(symmetry_index, tf.reshape(next_policy, [19, 19, 1])), [361])
    ownership = tf.reshape(_apply_symmetry(symmetry_index, tf.reshape(labels['ownership'], [19, 19, 1])), [361])

    labels['lz_features'] = lz_features
    labels['policy'] = tf.concat([policy, policy_pass], 0)
    labels['next_policy'] = tf.concat([next_policy, next_policy_pass], 0)
    labels['ownership'] = ownership

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


def get_dataset(files, is_deterministic=False):
    """ Returns a tf.DataSet initializable iterator over the given files """

    with tf.device('cpu:0'):
        num_parallel_calls = tf.data.experimental.AUTOTUNE if not is_deterministic else 1

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
        dataset = dataset.map(_parse(is_deterministic), num_parallel_calls=num_parallel_calls)
        dataset = dataset.filter(_legal_policy)

        return dataset


def input_fn(files, batch_size, features_mask, is_training, num_test_batches=10, is_deterministic=False):
    dataset = get_dataset(files, is_deterministic)

    if features_mask is not None:
        features_mask = tf.constant(features_mask, tf.float16, (1, 1, NUM_FEATURES))

        def _mask_features(features, labels):
            return features * features_mask, labels

        dataset = dataset.map(_mask_features)

    if is_training:
        num_parallel_calls = tf.data.experimental.AUTOTUNE if not is_deterministic else 1

        dataset = dataset.skip(num_test_batches * batch_size)
        dataset = dataset.shuffle(262144)
        dataset = dataset.repeat()
        dataset = dataset.map(_augment, num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(_fix_history, num_parallel_calls=num_parallel_calls)
    else:
        dataset = dataset.take(num_test_batches * batch_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
