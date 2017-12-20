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
import sys

def main():
    """ Main function """

    blacks = []  # queue of black wins that does not have a corresponding white win
    whites = []  # queue of white wins that does not have a corresponding black win
    unrecognized = 0

    for line in sys.stdin:
        assert not blacks or not whites

        if "RE[B" in line:
            blacks += (line,)
        elif "RE[W" in line:
            whites += (line,)
        else:
            unrecognized += 1

        if blacks and whites:
            print(blacks.pop().strip())
            print(whites.pop().strip())

    if unrecognized > 0:
        print('Discarded {} games without a winner'.format(unrecognized), file=sys.stderr)
    if blacks:
        print('Discarded {} black games'.format(len(blacks)), file=sys.stderr)
    if whites:
        print('Discarded {} white games'.format(len(whites)), file=sys.stderr)

if __name__ == '__main__':
    main()
