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
Convert coordinates in SGF files from _Sabaki_ format to _CGoban_ format.

Usage: ./sgf2cgoban.py < kgs_big.sgf > kgs_cgoban.sgf
"""
import re
import sys

VERTEX = re.compile(r';[BW]\[([a-z])([a-z])\]')
FLIP_TABLE = {
    'a': 's',
    'b': 'r',
    'c': 'q',
    'd': 'p',
    'e': 'o',
    'f': 'n',
    'g': 'm',
    'h': 'l',
    'i': 'k',
    'j': 'j',
    'k': 'i',
    'l': 'h',
    'm': 'g',
    'n': 'f',
    'o': 'e',
    'p': 'd',
    'q': 'c',
    'r': 'b',
    's': 'a',
    't': 't',
}

def flip_vertex(m):
    """ Reverse the y coordinate of the given vertex """
    x = m[1]
    y = FLIP_TABLE[m[2]]

    return m[0].replace(m[1] + m[2], x + y)

try:
    for line in sys.stdin:
        print(re.sub(VERTEX, flip_vertex, line))
except BrokenPipeError:
    pass
