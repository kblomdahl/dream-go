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
import fileinput as fi

from .ffi.libdg import parse_single_example, get_num_features

def _generate_random_value():
    x = tf.random.categorical(tf.log([[1.0, 1.0]]), 1)
    x = tf.one_hot(x, 2)

    return tf.reshape(x, [2])


def _generate_random_features():
    num_features = get_num_features()

    return tf.random.uniform([19, 19, num_features], 0.0, 1.0)


def _generate_random_policy():
    x = tf.random.categorical(tf.log([[1.0]*362]), 1)
    x = tf.one_hot(x, 362)

    return tf.reshape(x, [362])


def _generate_random_score():
    x = tf.random.categorical(tf.log([[1.0]*722]), 1)
    x = tf.one_hot(x, 722)

    return tf.reshape(x, [722])


def _generate_random_ownership():
    x = tf.random.categorical(tf.log([[1.0, 1.0]]), 361)
    x = tf.one_hot(x, 2)

    return tf.reshape(x, [19, 19, 2])


def random_input_fn(opts):
    data = tf.data.TextLineDataset(opts.files)

    def _to_dummy(line):
        return (
            _generate_random_features(),
            {
                'policy': _generate_random_policy(),
                'policy_next': _generate_random_policy(),
                'value': _generate_random_value(),
                'score': _generate_random_score(),
                'ownership': _generate_random_ownership(),
            }
        )

    return data \
        .map(_to_dummy) \
        .batch(opts.mini_batch_size) \
        .prefetch(4)


def _parse_single_line(line):
    """ Parse a single string containing an SGF file and return an example, or `None`
    if an error is encountered. """

    try:
        return parse_single_example(line.encode('utf-8'))
    except ValueError:
        return None
    except KeyboardInterrupt:
        return None


def _parse_files(files):
    """ Generator for all examples contained inside of the give files. This function
    will spin-up a pool of worker processors to do most of the actual parsing. """

    from multiprocessing import cpu_count, Pool

    num_processes = max(4, cpu_count() - 8)

    with fi.input(files=files) as f, Pool(num_processes) as p:
        for example in p.imap_unordered(_parse_single_line, f):
            if example:
                yield example


def file_input_fn(opts):
    num_features = get_num_features()
    data = tf.data.Dataset.from_generator(
        _parse_files,
        output_types=(
            tf.float32,
            {
                'policy': tf.float32,
                'policy_next': tf.float32,
                'value': tf.float32,
                'score': tf.float32,
                'ownership': tf.float32,
            }
        ),
        output_shapes=(
            tf.TensorShape([19, 19, num_features]),
            {
                'policy': tf.TensorShape([362]),
                'policy_next': tf.TensorShape([362]),
                'value': tf.TensorShape([2]),
                'score': tf.TensorShape([722]),
                'ownership': tf.TensorShape([19, 19, 2]),
            }
        ),
        args=[opts.files]
    )

    return data \
        .shuffle(262144) \
        .batch(opts.mini_batch_size) \
        .prefetch(4)