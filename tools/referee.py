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
GTP (Go Text Protocol) referee program for running engines against each other
some number of times to determine a winner.

Usage: ./referee.py [num games] <engines...>
"""

import sys
import subprocess
from math import sqrt
from random import random

import numpy as np

def gtp_to_sgf(vertex):
    """ Convert an GTP vertex (e.g. D4) to SGF format (dd) """
    GTP_LETTERS = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    SGF_LETTERS = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    x = GTP_LETTERS.index(vertex[0])
    y = int(vertex[1:]) - 1

    return '{}{}'.format(SGF_LETTERS[x], SGF_LETTERS[y])

def play_game(engine_1, engine_2):
    """
    Play a game of the two given engines and returns `0` if the first
    engine won, otherwise `1`.
    """
    proc_1 = subprocess.Popen(engine_1, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc_2 = subprocess.Popen(engine_2, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write_both(s):
        """ Write the given bytes to both of the engines. """
        proc_1.stdin.write(s)
        proc_2.stdin.write(s)

    def write_command(to, line_nr, s):
        """ Write a command to the given process and returns the response """
        to.stdin.write("{} {}".format(line_nr, s).encode('utf-8'))
        to.stdin.flush()

        for line in to.stdout:
            line = line.decode('utf-8').strip().lower()

            if line.startswith('=' + str(line_nr)):
                return line
            elif line.startswith('?' + str(line_nr)):
                print(line, file=sys.stderr)
                break

    try:
        write_both(b'komi 7.5\n')
        write_both(b'boardsize 19\n')
        write_both(b'clear_board\n')

        if random() < 0.5:
            black = (engine_1, proc_1, 'B', 0)
            white = (engine_2, proc_2, 'W', 1)
        else:
            black = (engine_2, proc_2, 'B', 1)
            white = (engine_1, proc_1, 'W', 0)

        turns = [black, white]
        all_moves = []
        pass_count = 0
        move_count = 0
        sgf = ''
        re = '?'

        while move_count < 722 and pass_count < 2:
            current, other = turns[0], turns[1]
            line_nr = move_count + 1000

            response = write_command(current[1], line_nr, 'genmove {}\n'.format(current[2]))
            if response:
                move = response.split(' ')[1]

                if move == 'pass':
                    pass_count += 1
                    sgf += ';{}[]'.format(current[2])
                elif move == 'resign':
                    re = '{}+Resign'.format(other[2])
                    return other[3]
                else:
                    pass_count = 0
                    all_moves.append((current[2], move))

                    # play this move to the other engine and then continue
                    # to its turn.
                    write_command(other[1], line_nr, 'play {} {}\n'.format(current[2], move))
                    sgf += ';{}[{}]'.format(current[2], gtp_to_sgf(move))
            else:
                # if an engine encounter an error, then it loses
                print('Received erroneous response from {}'.format(current[0]), file=sys.stderr)
                re = '{}+Resign'.format(other[2])
                return other[3]

            # flip the current player
            turns = [turns[1], turns[0]]
            move_count += 1

        proc_1.stdin.write(b'quit\n')
        proc_2.stdin.write(b'quit\n')

        # we finished without a winner, we use GNU Go to determine the winner so that
        # no complicated scoring algorihm, including figuring out if a group is alive
        # or not, has to be implemented in the referee.
        #
        # We use the `final_score` variant here to avoid any incorrect scores due to
        # complicated situations that `estimate_score` fail to take into account.
        gnugo = subprocess.Popen('gnugo --chinese-rules --mode gtp', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        gnugo.stdin.write(b'komi 7.5\n')
        gnugo.stdin.write(b'boardsize 19\n')
        gnugo.stdin.write(b'clear_board\n')

        for (i, (color, move)) in enumerate(all_moves):
            write_command(gnugo, 1000 + i, 'play {} {}\n'.format(color, move))

        re = write_command(gnugo, 2000, 'final_score\n')
        if not re:
            return  # cannot determine winner

        re = re.split(' ')[1].upper()  # trim away the id prefix

        gnugo.stdin.write(b'quit\n')
        gnugo.kill()
        gnugo.communicate()

        if re.startswith('b') or re.startswith('B'):
            return black[3]
        else:
            return white[3]
    finally:
        print('(;GM[1]FF[4]SZ[19]RU[Chinese]KM[7.5]PW[{}]PB[{}]RE[{}]{})'.format(
            white[0],
            black[0],
            re,
            sgf
        ), flush=True)

        # terminate the engines the hard way if they have not died already
        if not proc_1.poll():
            proc_1.kill()
            proc_1.communicate()
        if not proc_2.poll():
            proc_2.kill()
            proc_2.communicate()

def join_pretty(headers, elements):
    """
    Join the given elements with padding such that they align to the
    given headers.
    """

    max_header = max([len(header) for header in headers])
    num_padding = max_header + 2
    out = ''

    for element in elements:
        out += element
        out += ' ' * (num_padding - len(element))

    return out

def main(num_games, engines):
    """ Main function """

    # generate all unique pairings of engines
    num_engines = len(engines)
    pairings = []

    for i in range(num_engines):
        for j in range(i + 1, num_engines):
            pairings.append((i, j))

    # keep track of the number of times, for each `(engine_1, engine_2)`
    # pair, how many times `engine_1` has won.
    wins = np.zeros((num_engines, num_engines), np.int32)

    try:
        def total_played(engine_1, engine_2):
            """
            Returns the total number of games that the two given engines
            has played against each other.
            """
            return wins[engine_1, engine_2] + wins[engine_2, engine_1]

        while min([total_played(p[0], p[1]) for p in pairings]) < num_games:
            next_pair = min(pairings, key=lambda p: total_played(p[0], p[1]))
            winner = play_game(engines[next_pair[0]], engines[next_pair[1]])

            if winner == 0:
                wins[next_pair[0], next_pair[1]] += 1
            elif winner == 1:
                wins[next_pair[1], next_pair[0]] += 1
            else:
                pass  # error? retry again later
    finally:
        # pretty-print the engines scores, sorted according to their total number
        # of wins.
        print('# Played {} games between all engines, outputting the confidence'.format(np.sum(wins)), file=sys.stderr)
        print('# interval for each engine winning against every other engine.', file=sys.stderr)
        print(file=sys.stderr)

        names = ['{}. {}'.format(i + 1, engines[i]) for i in range(num_engines)]
        titles = ['vs {}'.format(name) for name in names]
        max_name = max([len(name) for name in names])
        name_padding = max_name + 4

        def get_score(engine_1, engine_2):
            """ Returns the score of `engine_1` vs `engine_2` """
            if engine_1 == engine_2:
                return '-'
            else:
                n_s = float(wins[engine_1, engine_2])
                n_f = float(wins[engine_2, engine_1])

                return '{} - {}'.format(int(n_s), int(n_f))

        print(' ' * name_padding + '{}'.format(join_pretty(titles, titles)), file=sys.stderr)
        for i in sorted(range(num_engines), key=lambda e: -np.sum(wins[e, :])):
            scores = [get_score(i, other) for other in range(num_engines)]

            print('{}{}{}'.format(
                names[i],
                ' ' * (name_padding - len(names[i])),
                join_pretty(titles, scores)
            ), file=sys.stderr)

if __name__ == '__main__':
    if len(sys.argv) <= 3:
        print('Usage: referee.py [num games] <engines...>')
        quit()

    main(int(sys.argv[1]), sys.argv[2:])
