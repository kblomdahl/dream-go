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

dream_go_module = tf.load_op_library('libdg_tf.so')

def _parse(num_unrolls):
    def __pad_to_channels(tensor, to_channels):
        n = tensor.shape.as_list()[-1]

        if to_channels == n:
            return tensor
        else:
            return tf.pad(
                tensor,
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, to_channels - n],
                ]
            )

    def __do_parse(line):
        features, motion_features, lz_features, targets, targets_mask, policy, value = dream_go_module.sgf_to_features(line, 2, num_unrolls)
        num_channels = max(features.shape.as_list()[-1], motion_features.shape.as_list()[-1])

        combined_features = tf.concat(
            [
                __pad_to_channels(features[:, :1, :, :, :], num_channels),
                __pad_to_channels(motion_features[:, 1:, :, :, :], num_channels),
            ],
            axis=1
        )

        labels = {
            'features': features,
            'motion_features': motion_features,
            'lz_features': lz_features,
            'targets': targets,
            'targets_mask': targets_mask,
            'value': value,
            'policy': policy
        }

        return tf.data.Dataset.from_tensor_slices(
            (combined_features, labels)
        )

    return __do_parse

def _legal_policy(features, labels):
    return tf.greater(tf.reduce_sum(input_tensor=labels['policy']), 0.0)

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
        dataset = dataset.interleave(
            _parse(num_unrolls),
            num_parallel_calls=num_parallel_calls,
            deterministic=False
        )
        dataset = dataset.filter(_legal_policy)

        return dataset

def input_fn(files, batch_size, is_training, *, num_unrolls=1, num_test_batches=10):
    dataset = get_dataset(files, num_unrolls=num_unrolls)

    if is_training is True:
        dataset = dataset.skip(num_test_batches * batch_size)
        dataset = dataset.shuffle(262144 // num_unrolls)
    elif is_training is False:
        dataset = dataset.take(num_test_batches * batch_size)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
