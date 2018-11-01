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

from subprocess import Popen, PIPE, DEVNULL
from random import random

import sys
import os

def play_game(engine_1, engine_2, environ_1, environ_2):
    """
    Play a game of the two given engines and returns `W` if the first
    engine won, otherwise `L`.
    """
    proc_1 = Popen(engine_1, shell=True, stdin=PIPE, stdout=PIPE, stderr=DEVNULL, env=environ_1)
    proc_2 = Popen(engine_2, shell=True, stdin=PIPE, stdout=PIPE, stderr=DEVNULL, env=environ_2)

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
                break

        return None

    try:
        write_both(b'komi 7.5\n')
        write_both(b'boardsize 19\n')
        write_both(b'clear_board\n')

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

            response = write_command(current[0], line_nr, 'genmove {}\n'.format(current[1]))
            if response:
                move = response.split(' ')[1]

                if move == 'pass':
                    pass_count += 1
                elif move == 'resign':
                    return other[2]  # if an engine resign then it loses
                else:
                    all_moves.append((current[1], move))
                    pass_count = 0

                    # play this move to the other engine
                    write_command(other[0], line_nr, 'play {} {}\n'.format(current[1], move))
            else:
                return other[2]  # if an engine encounter an error then it loses

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
        gnugo = Popen('/usr/games/gnugo --chinese-rules --mode gtp', shell=True, stdin=PIPE, stdout=PIPE, stderr=DEVNULL)
        gnugo.stdin.write(b'komi 7.5\n')
        gnugo.stdin.write(b'boardsize 19\n')
        gnugo.stdin.write(b'clear_board\n')

        for (i, (color, move)) in enumerate(all_moves):
            write_command(gnugo, 1000 + i, 'play {} {}\n'.format(color, move))

        re = write_command(gnugo, 2000, 'final_score\n')
        if not re:
            return None  # cannot determine winner

        re = re.split(' ')[1].upper()  # trim away the id prefix

        gnugo.stdin.write(b'quit\n')
        gnugo.communicate()

        if re.startswith('b') or re.startswith('B'):
            return black[2]
        else:
            return white[2]
    finally:
        # terminate the engines the hard way if they have not died already
        if not proc_1.poll():
            proc_1.kill()
            proc_1.communicate()
        if not proc_2.poll():
            proc_2.kill()
            proc_2.communicate()


def main():
    """ Main function """

    # complement the current environment variable with the given parameters
    parameters = dict(os.environ)

    for i in range(3, len(sys.argv), 2):
        parameters[sys.argv[i]] = sys.argv[i+1]

    #
    cwd = os.getcwd()

    print(play_game(
        cwd + "/../../target/release/dream_go --no-ponder --num-rollout 3200",  # engine to be optimized
        cwd + "/../../target/release/dream_go --no-ponder --num-rollout 3200",  # opponent
        parameters,
        {}
    ))

main()
