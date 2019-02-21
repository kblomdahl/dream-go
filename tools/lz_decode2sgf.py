#!/usr/bin/env python3
# Copyright 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
Converts leela-zero training data to Dream Go readable training data.

Usage: unzip -p train_ffe8ba44.zip | gunzip | ./lz_decode2sgf.py > lz_ffe8ba44_big.sgf
"""

import base64
import numpy as np
import sys
import re

WHITESPACE = re.compile("\s+")

def opponent(to_move):
    if to_move == 'B':
        return 'W'
    else:
        return 'B'

def interweave(a, b):
    for i in range(max(len(a), len(b))):
        if i < len(a):
            yield a[i]
        if i < len(b):
            yield b[i]

class Example:
    def __init__(self):
        self.to_move = None
        self.features = np.zeros((16, 361), '?')
        self.policy = np.zeros((362,), 'f4')
        self.value = 0.0

    def parse_feature_plane(self, line, feature_num):
        if feature_num == 16:
            # the 17th feature is current color to play, it is compacted to '0' for black
            self.to_move = 'B' if line == '0' else 'W'
        else:
            # Feature 1 ->  8: Players vertices
            # Feature 9 -> 16: Opponents vertices
            self.features[feature_num, :] = np.unpackbits(bytearray.fromhex(line + '0'))[:361]

    def parse_policy(self, line):
        self.policy[:] = [float(elem) for elem in re.split(WHITESPACE, line)]

    def parse_value(self, line):
        self.value = float(line)

    def get_feature_presence(self, index):
        """ Returns the number of features that the given _index_ is set in """
        for i in range(15, -1, -1):
            if self.features[i, index]:
                return i
        return 0

    @staticmethod
    def get_vertices_from(to_move, features, prev_features):
        for y in range(19):
            for x in range(19):
                index = 19 * y + x

                if features[index] and not prev_features[index]:
                    vertex = chr(97 + x) + chr(97 + y)

                    yield ';{}[{}]'.format(to_move, vertex)

    def get_prefix(self):
        # re-construct the game into an SGF from the history planes
        out = '(;GM[1]FF[4]SZ[19]RU[Chinese]KM[7.5]RE[{}+R]'.format(self.to_move if self.value > 0 else opponent(self.to_move))

        for i in range(8):
            black = tuple(Example.get_vertices_from(self.to_move, self.features[7-i, :], self.features[8-i, :] if i > 0 else np.zeros((361,), '?')))
            white = tuple(Example.get_vertices_from(opponent(self.to_move), self.features[15-i, :], self.features[16-i, :] if i > 0 else np.zeros((361,), '?')))

            if self.to_move != 'B':
                black, white = white, black

            for vertex in interweave(black, white):
                out += vertex

        return out

    def is_continuation(self, prev_example):
        """ Returns true if the number of new stones for this position, compared to
        the previous example, is one. """

        if prev_example is None:
            return False

        if self.to_move == prev_example.to_move:
            to_move_offset, opponent_offset = 0, 8
        else:
            to_move_offset, opponent_offset = 8, 0

        for i in range(7):
            if np.any(self.features[1+i, :] != prev_example.features[to_move_offset+i, :]):
                return False
            if np.any(self.features[9+i, :] != prev_example.features[opponent_offset+i, :]):
                return False

        return True

    def get_continuation(self, prev_example):
        if self.to_move == prev_example.to_move:
            to_move_offset, opponent_offset = 0, 8
        else:
            to_move_offset, opponent_offset = 8, 0

        black = tuple(Example.get_vertices_from(self.to_move, self.features[0, :], prev_example.features[to_move_offset, :]))
        white = tuple(Example.get_vertices_from(opponent(self.to_move), self.features[8, :], prev_example.features[opponent_offset, :]))
        out = ''

        if self.to_move != 'B':
            black, white = white, black

        for vertex in interweave(black, white):
            out += vertex

        return out + 'P[{}]'.format(base64.b85encode(np.asarray(prev_example.policy, 'f2').tostring(), pad=True).decode('ascii'))


def main():
    current_example, prev_example = Example(), None
    current_state = 0
    current_sgf = ''

    for line in sys.stdin:
        line = line.rstrip().lower()

        # Syntax:
        #    - 16x feature planes (hex encoded)
        #    - 1x  policy
        #    - 1x  value
        if current_state <= 16:
            current_example.parse_feature_plane(line, current_state)
        elif current_state == 17:
            current_example.parse_policy(line)
        elif current_state == 18:
            current_example.parse_value(line)

            if not current_example.is_continuation(prev_example):
                if current_sgf:
                    print(current_sgf + ')', flush=True)

                current_sgf = current_example.get_prefix()
            else:
                current_sgf += current_example.get_continuation(prev_example)

            current_example, prev_example = Example(), current_example

        current_state = (current_state + 1) % 19

    print(current_sgf + ')', flush=True)

main()
