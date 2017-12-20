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
Reads a _big_ SGF file from standard input and writes to standard output a sub-set
of the read SGF files that is perfectly balanced between black and white wins.

Usage: ./sgf2balance.py < kgs_big.sgf > kgs_big_bal.sgf
"""
from subprocess import Popen, PIPE, DEVNULL
import sys
import tempfile

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
            'gnugo --score aftermath --chinese-rules -l "' + sgf_file.name + '"',
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

def main():
    """ Main function """

    num_scored = 0

    for line in sys.stdin:
        line = line.strip()

        if "RE[?]" in line:
            winner = score_game(line)

            if winner:
                print(line.replace('RE[?]', 'RE[' + winner + ']'))
                num_scored += 1
            else:
                print(line)
        else:
            print(line)

    print('Arbitrated {} games without a winner'.format(num_scored), file=sys.stderr)

if __name__ == '__main__':
    main()
