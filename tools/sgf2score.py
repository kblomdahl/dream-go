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
Reads a _big_ SGF file from standard input and writes to standard output the
same games after they have been scored by a referee.

Usage: ./sgf2score.py < kgs_big.sgf > kgs_score.sgf
"""
import re
from subprocess import Popen, PIPE, DEVNULL
import sys
import tempfile

RE = re.compile(r'RE\[([^\]]+)\]')

def score_game(sgf):
    """
    Returns the winner of the game in the given SGF file as
    judged by `gnugo`.
    """

    with tempfile.NamedTemporaryFile() as sgf_file:
        sgf_file.write(sgf.encode())
        sgf_file.flush()

        # start-up our judge (gnugo)
        gnugo = Popen(
            '/usr/games/gnugo --score aftermath --chinese-rules --positional-superko -l "' + sgf_file.name + '"',
            shell=True,
            stdin=DEVNULL,
            stdout=PIPE,
            stderr=DEVNULL
        )

        for line in gnugo.stdout:
            line = line.decode('utf-8').strip()

            if 'White wins by' in line:  # White wins by 8.5 points
                return 'W+' + line.split()[3]
            elif 'Black wins by' in line:  # Black wins by 32.5 points
                return 'B+' + line.split()[3]

def main(every):
    """ Main function """

    num_scored = 0
    num_wrong = 0

    for line in sys.stdin:
        line = line.strip()

        if "RE[?]" in line:
            winner = score_game(line)

            if winner:
                print(re.sub(RE, 'RE[' + winner + ']', line))
                num_scored += 1
            else:
                print(line)
        else:
            winner = RE.search(line)

            if every and winner and 'R' not in winner.group(1).upper():
                winner = score_game(line)

                if winner[:4] not in line:
                    num_wrong += 1
                line = re.sub(RE, 'RE[' + winner + ']', line)

            print(line)

    if num_scored > 0:
        print('Arbitrated {} games without a winner'.format(num_scored), file=sys.stderr)
    if num_wrong > 0:
        print('Arbitrated {} games with the wrong winner'.format(num_wrong), file=sys.stderr)

if __name__ == '__main__':
    main(every=any([arg for arg in sys.argv if arg == '--all']))
