#!/usr/bin/env python3
# Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0103, C0301

"""
Arguments are:
1. processor id (symbolic name, typically a machine name to ssh to)
2. seed (integer)
3. parameter id of first parameter (symbolic name)
4. value of first parameter (float)
5. parameter id of second parameter (optional)
6. value of second parameter (optional)
...

This script should write the game outcome to its output:

* W = win
* L = loss
* D = draw

For instance:
```
$ ./DummyScript.py node-01 4 param 0.2
W
```
"""

from datetime import datetime
from multiprocessing import Pool, Queue
from subprocess import Popen, PIPE, DEVNULL, STDOUT
from random import random
import re

import sys
import os

LOG = open('run_dreamgo.log', 'a')

GTP_TABLE = {
    'a': 'a',
    'b': 'b',
    'c': 'c',
    'd': 'd',
    'e': 'e',
    'f': 'f',
    'g': 'g',
    'h': 'h',
    'j': 'i',
    'k': 'j',
    'l': 'k',
    'm': 'l',
    'n': 'm',
    'o': 'n',
    'p': 'o',
    'q': 'p',
    'r': 'q',
    's': 'r',
    't': 's',

    '1': 'a',
    '2': 'b',
    '3': 'c',
    '4': 'd',
    '5': 'e',
    '6': 'f',
    '7': 'g',
    '8': 'h',
    '9': 'i',
    '10': 'j',
    '11': 'k',
    '12': 'l',
    '13': 'm',
    '14': 'n',
    '15': 'o',
    '16': 'p',
    '17': 'q',
    '18': 'r',
    '19': 's',
}

def gtp_to_sgf(color, gtp_move):
    x = gtp_move[0]
    y = gtp_move[1:]

    return ';{}[{}{}]'.format(color, GTP_TABLE[x], GTP_TABLE[y])


def write_command(to, line_nr, s):
    """ Write a command to the given process and returns the response """
    to.stdin.write("{} {}".format(line_nr, s))
    to.stdin.flush()

    for original_line in to.stdout:
        print(original_line.rstrip(), file=LOG, flush=True)
        line = original_line.strip().lower()

        if line.startswith('=' + str(line_nr)):
            return line.split(' ', maxsplit=2)[-1].strip()
        elif line.startswith('?' + str(line_nr)):
            break

    return None


def play_game(proc_1, proc_2):
    """
    Play a game of the two given engines and returns `W` if the first
    engine won, otherwise `L`.
    """
    def write_both(s):
        """ Write the given bytes to both of the engines. """
        write_command(proc_1, 1000, s)
        write_command(proc_2, 1000, s)

    write_both('komi 7.5\n')
    write_both('boardsize 19\n')
    write_both('clear_board\n')

    if random() < 0.5:
        black = (proc_1, 'B', 'W')
        white = (proc_2, 'W', 'L')
    else:
        black = (proc_2, 'B', 'L')
        white = (proc_1, 'W', 'W')

    turns = [black, white]
    all_moves = []
    pass_count = 0
    move_count = 0

    while move_count < 722 and pass_count < 2:
        current, other = turns[0], turns[1]
        line_nr = move_count + 1000

        move = write_command(current[0], line_nr, 'genmove {}\n'.format(current[1]))
        if move:
            if move == 'pass':
                pass_count += 1
            elif move == 'resign':
                # if an engine resign then it loses
                return all_moves, (other[1] + '+resign'), black[2], white[2]
            else:
                all_moves.append((current[1], move))
                pass_count = 0

                # play this move to the other engine
                write_command(other[0], line_nr, 'play {} {}\n'.format(current[1], move))
        else:
            # if an engine encounter an error then it loses
            return all_moves, (other[1] + '+resign'), black[2], white[2]

        # flip the current player
        turns = [turns[1], turns[0]]
        move_count += 1

    return all_moves, None, black[2], white[2]


def score_game(all_moves, black_result, white_result):
    """ We use GNU Go to determine the winner so that no complicated scoring algorithm,
    including figuring out if a group is alive or not, has to be implemented in
    the referee.

    We use the `final_score` variant here to avoid any incorrect scores due to
    complicated situations that `estimate_score` fail to take into account.
    """
    gnugo = Popen([
        '/usr/games/gnugo',
        '--chinese-rules',
        '--positional-superko',
        '--mode', 'gtp'
    ], stdin=PIPE, stdout=PIPE, stderr=DEVNULL, encoding='utf-8')

    gnugo.stdin.write('komi 7.5\n')
    gnugo.stdin.write('boardsize 19\n')
    gnugo.stdin.write('clear_board\n')

    for (i, (color, move)) in enumerate(all_moves):
        write_command(gnugo, 1000 + i, 'play {} {}\n'.format(color, move))

    re = write_command(gnugo, 2000, 'final_score\n')
    if not re:
        return None, None, None  # cannot determine winner

    gnugo.stdin.write('quit\n')
    gnugo.communicate()

    return judge_result(re, black_result, white_result), re, all_moves

def judge_result(re, black_result, white_result):
    if re.startswith('b') or re.startswith('B'):
        return black_result
    else:
        return white_result

def main():
    """ Main function """

    # complement the current environment variable with the given parameters
    environ = dict(os.environ)
    extra = {}

    for i in range(3, len(sys.argv), 2):
        key = sys.argv[i]
        is_intp = re.match(r'([a-zA-Z_]+)\[([0-9]+)\]', key)

        if is_intp:
            extra.setdefault(is_intp[1], []).append('{},{}'.format(is_intp[2], sys.argv[i+1]))
        else:
            extra[key] = sys.argv[i+1]

    extra = { key: ':'.join(value) if isinstance(value, list) else value for key, value in extra.items() }

    cwd = os.getcwd()
    proc_1 = Popen(cwd + '/bin/dream_go --num-rollout 1600', shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, env={**environ, **extra}, encoding='utf-8')
    proc_2 = Popen(cwd + '/bin/dream_go --num-rollout 1600', shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, env=environ, encoding='utf-8')

    try:
        # play an odd number of games, and whomever wins the most is considered
        # the overall winner
        num_samples = 5
        results = {}

        with Pool(processes=num_samples) as pool:
            def accumulate_winner(args):
                """ Accumulate the winner of each game into `results` """
                result, winner, all_moves = args

                if result is not None:
                    results[result] = results.get(result, 0) + 1

                # save the final game as SGF
                if all_moves:
                    filename = datetime.now().strftime('%Y%m%d.%H%M%S.%f')

                    with open('games/' + filename + '.' + result + '.sgf', 'w') as sgf:
                        sgf.write('(;GM[1]FF[4]SZ[19]RU[Chinese]KM[7.5]')
                        sgf.write('RE[{}]'.format(winner))
                        sgf.write('C[{}\n{}]'.format(result, str(extra)))

                        for color, move in all_moves:
                            sgf.write(gtp_to_sgf(color, move))

                        sgf.write(')\n')

            for _ in range(num_samples):
                # early exit if any one engine has already won
                if len(results) >= num_samples/2 and max(results.values()) >= num_samples/2:
                    break

                moves, verdict, black_result, white_result = play_game(proc_1, proc_2)

                if not verdict:
                    pool.apply_async(score_game, (moves, black_result, white_result), callback=accumulate_winner)
                else:
                    accumulate_winner((judge_result(verdict, black_result, white_result), verdict, moves))

            write_command(proc_1, '1', 'quit\n')
            write_command(proc_2, '1', 'quit\n')

            pool.close()
            pool.join()

        print(max(results.keys(), key=lambda key: results[key]))
    finally:
        # terminate the engines the hard way if they have not died already
        if not proc_1.poll():
            proc_1.kill()
            proc_1.communicate()
        if not proc_2.poll():
            proc_2.kill()
            proc_2.communicate()

        LOG.close()
        LOG.flush()


main()
