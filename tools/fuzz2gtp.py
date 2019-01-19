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
Super simple fuzzing tool for GTP clients, and generate _self-play_ games forever
using the GTP interface.

Usage: ./fuzz2gtp.py ./dream_go --gtp
"""

from random import random, choice
from subprocess import Popen, PIPE, STDOUT
from sys import argv


def clear_board(proc):
    """ Sends the `clear_board` command to the client """
    proc.stdin.write('1 clear_board\n')
    proc.stdin.write('2 komi 7.5\n')
    proc.stdin.flush()


LETTERS = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't')
DIGITS = tuple(range(1, 20))

def random_vertex():
    """ Return a random GTP vertex in the format `xYY`, where `x` is a letter
    and `YY` is a digit. """
    return '%s%s' % (choice(LETTERS), choice(DIGITS))


def genmove(proc, colour, pluck_random=True):
    """ Send either a `genmove` command to the client, or generate a random
    move until it is accepted by the client """
    if pluck_random and random() < 0.05:
        for _count in range(100):
            proc.stdin.write('1000 play %s %s\n' % (colour, random_vertex(),))
            proc.stdin.flush()

            for line in proc.stdout:
                line = (str(line) or '').strip()
                print(line)

                if line.startswith('=1000'):
                    vertex = line.split(' ', maxsplit=2)[-1].strip()
                    return vertex
                elif line.startswith('?1000'):
                    break

        return 'pass'
    else:
        proc.stdin.write('2000 genmove %s\n' % (colour,))
        proc.stdin.flush()

        for line in proc.stdout:
            line = (str(line) or '').strip()
            print(line)

            if line.startswith('=2000'):
                vertex = line.split(' ', maxsplit=2)[-1].strip()

                return vertex

        return None


def play_game(proc):
    current, pass_count = 'B', 0

    clear_board(proc)
    while pass_count < 2:
        result_black = genmove(proc, current)
        if result_black == 'resign':
            break
        elif result_black == 'pass':
            pass_count += 1
        else:
            pass_count = 0

        current = 'B' if current == 'W' else 'W'


with Popen(' '.join(argv[1:]), shell=True, stdin=PIPE, stdout=PIPE, stderr=None, encoding='utf-8') as proc:
    while True:
        play_game(proc)

