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
Reads all SGF files contained within the given directories and converts them to
a single "big sgf" that is printed to stdout.

Usage: ./sgf2big.py <directories...>
"""

import mmap
import sys
import os


def skip_ws(string, i):
    """
    Returns the first index, larger than or equal to i that does not
    contain a whitespace character in the specified string.
    """

    while i < len(string) and chr(string[i]).isspace():
        i += 1

    return i


def parse_sgf_content(contents, i):
    """
    Parse the given SGF [1] content beginning at the given index and print a one-line version to stdout.

    [1] http://www.red-bean.com/sgf/sgf4.html
    """

    i = skip_ws(contents, i)
    skip = False
    out = ''

    # GameTree   = "(" Sequence { GameTree } ")"
    if i < len(contents) and contents[i] == ord('('):
        i = skip_ws(contents, i + 1)  # skip (
        out += '('

        # Sequence   = Node { Node }
        # Node       = ";" { Property }
        while i < len(contents) and contents[i] == ord(';'):
            i = skip_ws(contents, i + 1)  # skip ;
            out += ';'

            # Property   = PropIdent PropValue { PropValue }
            while i < len(contents) and chr(contents[i]).isalpha():
                # PropIdent  = UcLetter { UcLetter }
                ident = ''

                while i < len(contents) and chr(contents[i]).isalpha():
                    ident += chr(contents[i])
                    i += 1

                if ident == 'AB':  # handicap
                    skip = True

                i = skip_ws(contents, i)

                if i < len(contents) and contents[i] != ord('['):
                    return

                i += 1

                # PropValue  = "[" CValueType "]"
                value = ''

                while i < len(contents) and (contents[i-1] == ord('\\') or contents[i] != ord(']')):
                    value += chr(contents[i])
                    i += 1

                i = skip_ws(contents, i + 1)

                # skip handicap games
                if ident == 'HA' and value != '0':
                    skip = True

                # skip comments
                if not skip and ident != 'C':
                    out += ident
                    out += '['
                    out += value
                    out += ']'

        i = skip_ws(contents, i + 1)  # skip )
        out += ')'
    else:
        return  # invalid sgf

    if not skip:
        print(out)
    return i


def parse_sgf(path):
    """
    Parse the given SGF [1] file and print a one-line version to stdout.

    [1] http://www.red-bean.com/sgf/sgf4.html
    """

    # Collection = GameTree { GameTree }
    # GameTree   = "(" Sequence { GameTree } ")"
    # Sequence   = Node { Node }
    # Node       = ";" { Property }
    # Property   = PropIdent PropValue { PropValue }
    # PropIdent  = UcLetter { UcLetter }
    # PropValue  = "[" CValueType "]"

    with open(path, 'rb') as f:
        contents = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        current_position = 0

        while current_position is not None and current_position < len(contents):
            current_position = parse_sgf_content(contents, current_position)


def main(base_dir):
    """ Parse and output all SGF files contained within the given directory """

    for root, _, files in os.walk(base_dir):
        for name in files:
            path = os.path.join(root, name)

            parse_sgf(path)


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        main(arg)
