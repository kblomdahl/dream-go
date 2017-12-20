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
Tool for converting an SGF file to Rust arrays of triplet (color, x, y)
for easier embeddings into unit tests.

Usage: ./sgf2array [number of moves] < input.sgf
"""
import re
import sys

# The letters used to describe board coordinates in an SGF file
LETTERS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

def main(max_count):
    """ Main function """

    elements = []

    for line in sys.stdin:
        for match in re.finditer(r"([BW])\[([a-z]*)\]", line):
            if match.group(2):  # not pass
                x = LETTERS.index(match.group(2)[0])
                y = LETTERS.index(match.group(2)[1])

                if match.group(1) == 'B':
                    elements.append(('Color::Black', x, y))
                elif match.group(1) == 'W':
                    elements.append(('Color::White', x, y))

    # pretty-print all of the elements
    count = 0

    for (color, x, y) in elements:
        if count > 0 and count % 4 == 0:
            print(',')
        elif count > 0:
            print(', ', end='')

        print('({:s}, {:2d}, {:2d})'.format(color, x, y), end='')

        count += 1
        if count >= max_count:
            break

    print('', flush=True)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main(100000)
