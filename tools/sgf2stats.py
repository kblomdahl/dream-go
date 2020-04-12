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

Usage: ./sgf2stats.py < kgs_big.sgf
"""
from collections import defaultdict
from functools import reduce
import re
import sys
import math

VERTEX = re.compile(r';([BW])\[([a-z][a-z])?\]')
LETTERS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

def pretty_print_array(array, variable_name):
    print('', flush=False)
    print(f'{variable_name} = [')
    for (i, h) in enumerate(array):
        if h == -h:
            h = abs(h)
        print(f'{h:.5e},', end='', flush=False)

        if (i + 1) % 6 == 0:
            print('', flush=False)
        else:
            print(' ', end='', flush=False)

    print('', flush=True)
    print(']', flush=True)


def text_histogram(values, title, bucket_size=10):
    buckets = {}

    for value in values:
        value_down = value - value % bucket_size
        buckets[value_down] = buckets.setdefault(value_down, 0) + 1

    for key, value in buckets.items():
        buckets[key] = value - min(buckets.values())

    while max(buckets.values()) > 60:
        for key, value in buckets.items():
            buckets[key] = value // 2

    print(f'------+-{"-" * max(buckets.values())}')
    print(f'      | {title}')
    print(f'------+-{"-" * max(buckets.values())}')
    for key in sorted(buckets.keys()):
        print(f'{key:5} | {"#" * buckets[key]}')
    print(f'------+-{"-" * max(buckets.values())}')


def vertex_to_index(vertex):
    if vertex == '':
        return 361
    else:
        y = LETTERS.index(vertex[0])
        x = LETTERS.index(vertex[1])

        return 19 * y + x


hashes_per_move = defaultdict(lambda: set())
total_per_move = {}
total_per_vertex = {}
num_moves_per_game = []

try:
    for line in sys.stdin:
        # do not actually play out the entire game, instead assume no stones will
        # ever get captures (obviously false, but _good enough_). This allows us
        # to distinguish board state purely by the played stone, if we discard their
        # order.
        moves_in_game = list(map(lambda m: m[2], re.finditer(VERTEX, line)))
        if not 'RE[B+R]' in line and not 'RE[W+R]' in line:
            moves_in_game += ['', '']

        num_moves_in_game = len(moves_in_game)
        zobrist_hash = 0

        for i in range(num_moves_in_game):
            vertex = moves_in_game[i]
            zobrist_hash ^= hash(vertex)

            hashes_per_move[i].add(zobrist_hash)
            total_per_move[i] = total_per_move.setdefault(i, 0) + 1
            total_per_vertex[vertex] = total_per_vertex.setdefault(vertex, 0) + 1

        num_moves_per_game.append(num_moves_in_game)
except BrokenPipeError:
    pass

# random fun statistics
text_histogram(num_moves_per_game, 'Number of moves per game')

# pretty-print the policy offsets
total_moves = sum([n for n in total_per_vertex.values()])
total_per_index = [1.0]*362
for (vertex, v) in total_per_vertex.items():
    total_per_index[vertex_to_index(vertex)] = math.log(v / total_moves)

pretty_print_array(total_per_index, 'POLICY_OFFSET')

# pretty-print the board diversity
cumulative_num_hashes_per_move = {}
for (i, hashes) in hashes_per_move.items():
    cumulative_num_hashes_per_move[i] = len(hashes)
    if i > 0:
        cumulative_num_hashes_per_move[i] += cumulative_num_hashes_per_move[i-1]

total_num_hashes = max(cumulative_num_hashes_per_move)
pct_hashes_per_move = [cumulative_num_hashes_per_move[i] / total_num_hashes for i in sorted(cumulative_num_hashes_per_move.keys())]
normalized_hashes_per_move = [pct / max(pct_hashes_per_move) for pct in pct_hashes_per_move]

pretty_print_array(normalized_hashes_per_move, 'VALUE_BOOST')
