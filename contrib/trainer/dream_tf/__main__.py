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

import argparse
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from .layers import set_compute_type
from .hooks.dump import DumpHook
from .hooks.learning_rate import LearningRateScheduler
from .input_fn import input_fn
from .model_fn import model_fn
from .sgf.heat_map import to_sgf_heat_map

""" The default total number of examples to train over """
MAX_STEPS = 524288000

""" The default number of examples per batch """
BATCH_SIZE = 2048


def parse_args():
    """ Returns an `argparse` parser for dealing with the command-line
    arguments. """

    parser = argparse.ArgumentParser(
        prog='dream_tf',
        description='Neural network optimizer for Dream Go.'
    )

    parser.add_argument('files', nargs=argparse.REMAINDER, help='The binary features files.')

    opt_group = parser.add_argument_group(title='optional configuration')
    opt_group.add_argument('--batch-size', nargs=1, type=int, metavar='N', help='the number of examples per mini-batch')
    opt_group.add_argument('--warm-start', nargs=1, metavar='M', help='initialize weights from the given model')
    opt_group.add_argument('--steps', nargs=1, type=int, metavar='N', help='the total number of examples to train over')
    opt_group.add_argument('--model', nargs=1, help='the directory that contains the model')
    opt_group.add_argument('--name', nargs=1, help='the name of this session')
    opt_group.add_argument('--debug', action='store_true', help='enable command-line debugging')
    opt_group.add_argument('--deterministic', action='store_true', help='enable deterministic mode')

    opt_group = parser.add_argument_group(title='model configuration')
    opt_group.add_argument('--num-channels', nargs=1, type=int, metavar='N', help='the number of channels per residual block')
    opt_group.add_argument('--num-blocks', nargs=1, type=int, metavar='N', help='the number of residual blocks')
    opt_group.add_argument('--mask', nargs=1, metavar='M', help='mask to multiply features with')

    op_group = parser.add_mutually_exclusive_group(required=True)
    op_group.add_argument('--start', action='store_true', help='start training of a new model')
    op_group.add_argument('--resume', action='store_true', help='resume training of an existing model')
    op_group.add_argument('--verify', action='store_true', help='evaluate the accuracy of a model')
    op_group.add_argument('--dump', action='store_true', help='print the weights of a model to standard output')
    op_group.add_argument('--features-map', action='store_true', help='print the final tower features to standard output')
    op_group.add_argument('--print', action='store_true', help='print the value of the given tensor')

    return parser.parse_args()


def most_recent_model():
    """ Returns the directory in `models/` that is the most recent """
    import os

    all_models = ['models/' + m for m in os.listdir('models/')]

    return max(
        [m for m in all_models if os.path.isdir(m)],
        key=os.path.getmtime
    )


def get_num_channels(args, model_dir):
    """ Returns the number of channels to use when constructing the model. """
    if args.num_channels:
        return args.num_channels[0]

    try:
        return tf.train.load_variable(model_dir, 'num_channels')
    except tf.errors.NotFoundError:
        return None
    except tf.errors.InvalidArgumentError:
        return None


def get_num_blocks(args, model_dir):
    """ Returns the number of blocks to use when constructing the model. """
    if args.num_blocks:
        return args.num_blocks[0]

    try:
        return tf.train.load_variable(model_dir, 'num_blocks')
    except tf.errors.NotFoundError:
        return None
    except tf.errors.InvalidArgumentError:
        return None


def main():
    args = parse_args()
    model_dir = args.model[0] if args.model else None
    if not model_dir:
        if args.start:
            model_dir = 'models/' + datetime.now().strftime('%Y%m%d.%H%M')

            if args.name:
                model_dir += '-' + args.name[0] + '/'
            else:
                model_dir += '/'
        else:
            model_dir = most_recent_model()

    params = {
        'steps': args.steps[0] if args.steps else MAX_STEPS,
        'batch_size': args.batch_size[0] if args.batch_size else BATCH_SIZE,
        'learning_rate': 1e-4 if args.warm_start else 3e-4,

        'num_channels': get_num_channels(args, model_dir) or 192,
        'num_blocks': get_num_blocks(args, model_dir) or 16
    }

    config = tf.estimator.RunConfig(
        tf_random_seed=0xfde6885f if args.deterministic else None,
        session_config=tf.ConfigProto(
            graph_options=tf.GraphOptions(
                optimizer_options=tf.OptimizerOptions(
                    do_common_subexpression_elimination=not args.debug,
                    do_constant_folding=not args.debug,
                    do_function_inlining=not args.debug
                )
            ),
            gpu_options=tf.GPUOptions(
                allow_growth=True
            )
        )
    )

    if args.warm_start:
        steps_to_skip = 10000
        warm_start_from = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=args.warm_start[0],
            vars_to_warm_start='[0-9x].*'  # only layers
        )
    else:
        steps_to_skip = 0
        warm_start_from = None

    if args.deterministic:
        # since 16-bit floating point is not accurate enough for deterministic output, fix
        # it to `f32` instead.
        set_compute_type(tf.float32)

    if args.mask:
        features_mask = list(map(lambda x: float(x), args.mask[0].split(';')))
    else:
        features_mask = None

    hooks = [tf_debug.LocalCLIDebugHook()] if args.debug else []
    nn = tf.estimator.Estimator(
        config=config,
        model_fn=model_fn,
        model_dir=model_dir,
        params=params,
        warm_start_from=warm_start_from
    )

    if args.start or args.resume:
        nn.train(
            hooks=hooks + [LearningRateScheduler(steps_to_skip)],
            input_fn=lambda: input_fn(args.files, params['batch_size'], features_mask, True, args.deterministic),
            steps=params['steps'] // params['batch_size']
        )
    elif args.verify:
        # iterate over the entire dataset and collect the metric, which we will
        # then pretty-print as a JSON object to standard output
        results = nn.evaluate(
            hooks=hooks,
            input_fn=lambda: input_fn(args.files, params['batch_size'], features_mask, False, args.deterministic),
            steps=params['steps'] // params['batch_size']
        )

        print(json.dumps(
            results,
            default=lambda x: float(x) if x != int(x) else int(x),  # handle `Decimal` types
            sort_keys=True,
            separators=(',', ': '),
            indent=4
        ))
    elif args.dump:
        predictor = nn.predict(
            input_fn=lambda: input_fn([], params['batch_size'], None, False, False),
            hooks=[DumpHook()]
        )

        for _ in predictor:
            pass
    elif args.features_map > 0:
        predictor = nn.predict(
            input_fn=lambda: input_fn(args.files, 1, features_mask, False, args.deterministic)
        )
        count = 0

        print('(;GM[1]FF[4]SZ[19]')
        for results in predictor:
            board_state = to_sgf_heat_map(results['features'], results['tower'])

            print('(;{})'.format(board_state))

            count += 1
            if count > 100:
                break
        print(')')
    elif args.print:
        # tensors are given then print all available tensors with some statistics.
        if not args.files:
            out = {}

            for var in nn.get_variable_names():
                var_value = np.asarray(nn.get_variable_value(var))

                out[var] = {
                    'mean': float(np.average(var_value)),
                    'std': float(np.std(var_value))
                }

            print(json.dumps(
                out,
                default=lambda x: float(x) if x != int(x) else int(x),  # handle `Decimal` types
                sort_keys=True,
                separators=(',', ': '),
                indent=4
            ))
        else:
            for var in args.files:
                print(var, nn.get_variable_value(var).tolist())


if __name__ == '__main__':
    main()
