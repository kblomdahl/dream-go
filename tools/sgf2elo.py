#!/usr/bin/env python3
# Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
Usage: ./sgf2elo.py < games_big.sgf
"""

import re
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix, lil_matrix, triu
import sys

NAMES, INDICES = {}, {}
PB = re.compile(r'PB\[([^\]]*)\]')
PW = re.compile(r'PW\[([^\]]*)\]')
RE = re.compile(r'RE\[([BbWw])')

def to_idx(name):
    """ Returns an bijective index for the given name. """
    name = name.strip().lower()
    if name not in NAMES:
        NAMES[name] = len(NAMES)
        INDICES[NAMES[name]] = name

    return NAMES[name]

# collect the raw number of wins and played games from standard input
# using DOK matrices for fast random access and growth.
wins = dok_matrix((0, 0), 'i4')
count = dok_matrix((0, 0), 'i4')

for line in sys.stdin:
    s_pb, s_pw, s_re = PB.search(line), PW.search(line), RE.search(line)
    if not s_pb or not s_pw or not s_re:
        continue

    if s_re.group(1).upper() == 'B':
        winner, loser = s_pb.group(1), s_pw.group(1)
    else:
        winner, loser = s_pw.group(1), s_pb.group(1)

    i_w = to_idx(winner)
    i_l = to_idx(loser)
    i_m = max((i_w, i_l))

    if i_m >= wins.shape[0]:
        wins.resize((i_m + 1, i_m + 1))
        count.resize((i_m + 1, i_m + 1))

    wins[i_w, i_l] += 1
    count[i_w, i_l] += 1
    count[i_l, i_w] += 1

# compute the winrate and then convert the winrate to elo rating difference
nz = triu(count).nonzero()
winrate = wins[nz] / count[nz]
elo_diff = 400.0 * np.log(1.0 / np.asarray(winrate.data) - 1.0) / np.log(10.0)

# convert the elo difference to the linear equations
nr_constraints = len(nz[0])
C = np.zeros((nr_constraints, count.shape[0]))
b = np.zeros((nr_constraints,))

for row in range(nr_constraints):
    C[row, nz[0][row]] = -1
    C[row, nz[1][row]] = +1
    b[row] = elo_diff[0, row]

# find the best fit to those linear equations
y = np.linalg.lstsq(C, b)[0]
y_min = -np.min(y)

for i in np.argsort(y):
    print('{:24}\t{:8.2f}'.format(INDICES[i], y_min + y[i]))
