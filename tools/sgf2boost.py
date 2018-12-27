#!/usr/bin/env python3
# Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0103, C0301

"""
Calculate board diversity in SGF files, after `x` number of moves.

Usage: ./sgf2boost.py < kgs_big.sgf
"""
from functools import reduce
import re
import sys

VERTEX = re.compile(r';([BW])\[([a-z][a-z])?\]')

hashes_per_move = {}
total_per_move = {}
num_moves_per_game = []

try:
    for line in sys.stdin:
        # do not actually play out the entire game, instead assume no stones will
        # ever get captures (obviously false, but _good enough_). This allows us
        # to distinguish board state purely by the played stone, if we discard their
        # order.
        moves_in_game = list(map(lambda m: m[0], re.finditer(VERTEX, line)))
        num_moves_in_game = len(moves_in_game)

        for i in range(num_moves_in_game):
            moves_so_far = moves_in_game[:(i+1)]
            zobrist_hash = reduce(lambda x, y: x ^ y, [hash(v) for v in moves_so_far], 0)

            hashes_per_move.setdefault(i, set()).add(zobrist_hash)
            total_per_move[i] = total_per_move.setdefault(i, 0) + 1

        num_moves_per_game.append(num_moves_in_game)
except BrokenPipeError:
    pass

# random fun statistics (TODO: Make it an ascii histogram)
print('Average length of games:', sum(num_moves_per_game) / len(num_moves_per_game))

# pretty-print the array of weights
normalized_hashes_per_move = [len(h) / total_per_move[i] for (i, h) in hashes_per_move.items()]

print('', flush=False)
for (i, h) in enumerate(normalized_hashes_per_move):
    print('{:.5e},'.format(h), end='', flush=False)

    if (i + 1) % 6 == 0:
        print()
    else:
        print(' ', end='', flush=False)

print('', flush=True)

