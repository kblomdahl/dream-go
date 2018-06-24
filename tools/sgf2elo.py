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
from scipy.sparse import coo_matrix, dok_matrix, triu
import sys

NAMES, INDICES = {}, {}
PB = re.compile(r'PB\[([^\]]*)\]')
PW = re.compile(r'PW\[([^\]]*)\]')
RE = re.compile(r'RE\[([BbWw])')

def to_idx(name):
    """ Returns an bijective index larger than zero for the given name. """
    name = name.strip().lower()
    if name not in NAMES:
        NAMES[name] = len(NAMES)
        INDICES[NAMES[name]] = name

    return NAMES[name]

# collect the raw number of wins and played games from standard input
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

wins = wins.tocsc()
count = count.tocsc()

# remove perfect winners and losers as we do not have enough information
# to determine their rating (would be Inf and -Inf)
matches = count.nonzero()
perfect_winners = coo_matrix(count.shape, dtype='?')
perfect_losers = coo_matrix(count.shape, dtype='?')

for (i, j) in zip(matches[0], matches[1]):
    if wins[i, j] == count[i, j]:
        perfect_winners[i, j] = True
    elif wins[i, j] == 0:
        perfect_losers[i, j] = True

wins[perfect_winners] = 0
count[perfect_winners] = 0
count[perfect_losers] = 0

# only keep the upper triangle of the matrix since they lower triangle is
# redundant (`wins[lower] = count - wins[upper]`).
nr_participants = len(NAMES)
nz = triu(count).nonzero()

# find the best fit using the maximum log-likelihood function. We use the
# bionomial distribution is calculate the likelihood that each specific
# match-up receive the exact number of wins given some probability.
from scipy.optimize import minimize
from scipy.stats import binom

def likelihood(x):
    def elo(delta):
        return 1.0 / (1 + 10.0 ** (delta / 400.0))

    p_win = elo(x[nz[1]] - x[nz[0]])
    p = binom.logpmf(wins[nz], count[nz], p_win)

    assert (p_win < 1).all()  # check that we do not predict any perfect winners
    assert (p < 0).all()  # check that probability is between [0, 1)

    return -0.5 * np.sum(p)

result = minimize(
    likelihood,
    np.random.uniform(0.0, 1000.0, nr_participants)
)

y = result.x
y_min = np.min(y)

for i in np.argsort(y):
    print('{:24}\t{:8.2f}'.format(INDICES[i], y[i] - y_min))
