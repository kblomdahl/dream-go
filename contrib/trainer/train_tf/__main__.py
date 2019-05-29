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

import argparse
import os
import sys
import tensorflow as tf

from .callbacks.pretty_tensorboard import PrettyTensorBoard
from .serializer import serialize_to
from .input_fn import file_input_fn, random_input_fn
from .model_fn import model_fn
from .train_fn import fit

# global average pooling:
#   policy_accuracy: 0.42
#   policy_next_accuracy: 0.07
#   value_accuracy: 0.67
#
# squeeze excitation:
#   policy_accuracy: 0.43
#   policy_next_accuracy: 0.07
#   value_accuracy: 0.66
#

def sgf_file(string):
    """ Returns the given string if it is a readable file of game records. """

    try:
        with open(string, 'r') as file:
            for line in file:
                line = line.strip()

                if len(line) > 0 and not line.startswith('('):
                    raise argparse.ArgumentTypeError('cannot parse: {}'.format(string))

        return string
    except FileNotFoundError:
        raise argparse.ArgumentTypeError('cannot open: {}'.format(string))


def model_directory(string):
    """ Return the given string if it is a directory  """

    if not os.path.isdir(string):
        raise argparse.ArgumentTypeError('not a directory: {}'.format(string))
    return string


def session_name(string):
    """ Return the given string if it is a valid session name  """

    if any([ch not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for ch in string.lower()]):
        raise argparse.ArgumentTypeError('illegal session name: {}'.format(string))
    return string


def get_argument_parser():
    parser = argparse.ArgumentParser(
        prog='train_tf',
        description='Training script for dream_go'
    )

    parser.add_argument('files', type=sgf_file, nargs='+')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--start', dest='mode', action='store_const', const='start',
                            help='start a new training run')
    mode_group.add_argument('--resume', dest='mode', action='store_const', const='resume',
                            help='resume a previous training run')
    mode_group.add_argument('--test', dest='mode', action='store_const', const='test',
                            help='print the accuracy of a previous run to standard output')

    optimizer_group = parser.add_argument_group('optimization arguments')
    optimizer_group.add_argument('--batch-size', metavar='N', type=int, default=2048,
                                 help='the number of examples per optimization step')
    optimizer_group.add_argument('--mini-batch-size', metavar='N', type=int, default=128,
                                 help='the number of examples per backpropagation step')
    optimizer_group.add_argument('--lr', metavar='R', type=float, default=0.002,
                                 help='the learning rate')

    model_group = parser.add_argument_group('model arguments')
    model_group.add_argument('--num-blocks', metavar='N', type=int, default=6,
                             help='the number of residual blocks')
    model_group.add_argument('--num-channels', metavar='N', type=int, default=64,
                             help='the number of channels per residual block')

    other_group = parser.add_argument_group('other arguments')
    other_group.add_argument('--deterministic', action='store_true',
                             help='enable deterministic mode')
    other_group.add_argument('--profile', action='store_true',
                             help='enable profiling')
    other_group.add_argument('--model', metavar='M', type=model_directory,
                             help='the directory that contains the model')
    other_group.add_argument('--name', metavar='N', type=session_name,
                             help='the name of this session')

    return parser


def most_recent_model_directory():
    """ Returns the directory in `models/` that is the most recent """
    import os

    all_models = ['models/' + m for m in os.listdir('models/')]

    return max(
        [m for m in all_models if os.path.isdir(m)],
        key=os.path.getmtime
    )


def default_model_directory(opts):
    if opts.mode == 'start':
        from datetime import datetime

        model_dir = 'models/' + datetime.now().strftime('%Y%m%d.%H%M')

        if opts.name:
            model_dir += '-' + opts.name

        return model_dir
    else:
        return most_recent_model_directory()


def most_recent_weights_file(opts):
    """ Returns the weight file in `models/.../models` that is the most recent """
    import os

    all_weights = [opts.model + '/models/' + m for m in os.listdir(opts.model + '/models/')]
    all_weights = [weights_file for weights_file in all_weights if os.path.isfile(weights_file)]

    return max(
        [w for w in all_weights],
        key=os.path.getmtime
    ), len(all_weights)


def main(args):
    opts = get_argument_parser().parse_args(args)

    if not opts.model:
        opts.model = default_model_directory(opts)
    os.makedirs(opts.model + '/models/', exist_ok=True)

    #
    if opts.profile:
        tf.enable_eager_execution()

    # build the input pipeline and model
    data_generator = file_input_fn(opts)  # random_input_fn(opts)
    model = model_fn(opts, data_generator.output_shapes[0])

    # restore the most recent training weights
    try:
        weights_file, initial_epoch = most_recent_weights_file(opts)

        model.load_weights(weights_file, by_name=True)
    except ValueError:
        initial_epoch = 0

    if opts.mode in ['start', 'resume']:
        def on_epoch_end(epoch, logs=None):
            filename = opts.model + '/dream_go.json'

            with open(filename, 'w') as fp:
                serialize_to(
                    {"features": model.input},
                    {name: output for name, output in zip(model.output_names, model.outputs)},
                    fp
                )

        model.fit(
            data_generator,
            initial_epoch=initial_epoch,
            epochs=500,
            shuffle=not opts.deterministic,
            callbacks=[
                tf.keras.callbacks.TerminateOnNaN(),
                #tf.keras.callbacks.EarlyStopping('loss', patience=2, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(opts.model + '/models/model.{epoch:04d}.hdf5'),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end),
                PrettyTensorBoard(opts.model),
            ]
        )


if __name__ == '__main__':
    main(sys.argv[1:])
