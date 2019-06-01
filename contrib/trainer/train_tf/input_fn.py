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

import itertools
import shlex
import tensorflow as tf

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
    x = tf.random.categorical(tf.log([[1.0]*723]), 1)
    x = tf.one_hot(x, 723)

    return tf.reshape(x, [723])


def _generate_random_ownership():
    x = tf.random.categorical(tf.log([[1.0, 1.0]]), 361)
    x = tf.one_hot(x, 2)

    return tf.reshape(x, [19, 19, 2])


def random_input_fn(opts):
    data = tf.data.TextLineDataset(opts.files)

    def _to_dummy(_line):
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
        .repeat() \
        .map(_to_dummy) \
        .batch(opts.mini_batch_size) \
        .prefetch(4)


def _parse_single_line(line):
    """ Parse a single string containing an SGF file and return an example, or `None`
    if an error is encountered. """

    try:
        return parse_single_example(line)
    except ValueError:
        return None
    except KeyboardInterrupt:
        return None


def _get_num_lines(file):
    """ Returns the number of lines in the given file """

    with open(file, 'r') as f:
        return len(f.readlines())


def _yield_from_sh(command):
    import subprocess

    with subprocess.Popen(
            command,
            shell=True,
            close_fds=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
    ) as proc:
        for line in proc.stdout:
            yield line


def _head(file, n):
    """ Returns the first `n` lines from the given file, if `n` is negative
    then all except the last `-n` lines are generated. """

    yield from _yield_from_sh(
        "head -n {} {} | shuf".format(
            int(n),
            shlex.quote(file.decode())
        )
    )


def _tail(file, n):
    """ Returns the last `n` lines from the given file, if `n` is negative
    then all except the first `-n` lines are generated. """

    yield from _yield_from_sh(
        "tail -n {} {}".format(
            int(n),
            shlex.quote(file.decode())
        )
    )


def _parse_files(files, is_training):
    """ Generator for all examples contained inside of the give files. This function
    will spin-up a pool of worker processors to do most of the actual parsing. """

    lines_per_file = {file: _get_num_lines(file) for file in files}
    all_lines = [
        _head(file, -num_lines // 20) if is_training else _tail(file, num_lines // 20)
        for
        file, num_lines in lines_per_file.items()
    ]

    # parse the SGF files and extract the features in a background worker pool to
    # ensure they can be done in parallel with the main training loop
    from multiprocessing import cpu_count, Pool

    num_processes = max(4, cpu_count() - 8)

    with Pool(num_processes) as p:
        lines = itertools.chain.from_iterable(all_lines)

        for example in p.imap_unordered(_parse_single_line, lines):
            if example:
                yield example


def file_input_fn(opts, is_training=True):
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
                'score': tf.TensorShape([723]),
                'ownership': tf.TensorShape([19, 19, 2]),
            }
        ),
        args=[opts.files, is_training]
    )

    if is_training:
        data = data.repeat()

    return data \
        .batch(opts.mini_batch_size) \
        .prefetch(4)
