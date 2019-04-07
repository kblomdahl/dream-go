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
of the read SGF file that contains the _k_ most recent games:

Usage: ./sgf2recent.py [number|date] < kgs_big.sgf > kgs_big_recent.sgf
"""
from datetime import datetime
import heapq
import sys
import re

DT = re.compile(r'DT\[([^\]]*)\]')
DT_FORMATS = [
    '%Y-%m-%dT%H:%M:%S%z',  # internal dream go
    '%Y.%m.%d %H:%M',  # tygem
    '%Y-%m-%d',  # standard sgf
]


def str_to_time(s):
    """ Try to parse the given string as a datetime object """
    d = None

    for dt_format in DT_FORMATS:
        try:
            d = datetime.strptime(s, dt_format)
            break
        except ValueError:
            pass

    return d


def sgf_to_time(line):
    """ Returns the creation time of the given SGF file """
    m = DT.search(line)

    if m:
        return str_to_time(m.group(1))
    else:
        return None

def dump_most_recent(k):
    """ Print the `k` most recent games from standard input """

    # collect all lines into a min-heap, discarding the minimum
    # element every time it grows too large
    entries = []
    unrecognized = 0
    count = 0

    for line in sys.stdin:
        line = line.strip()
        d = sgf_to_time(line)

        if not d:
            unrecognized += 1
        else:
            item = (d, line)

            if len(entries) > k:
                heapq.heapreplace(entries, item)
            else:
                heapq.heappush(entries, item)

        count += 1

    # print each entry, from the oldest to the most recent:
    if len(entries) > k:
        heapq.heappop(entries)  # discard the buffer element

    for (_, line) in [heapq.heappop(entries) for _ in range(len(entries))]:
        print(line)

    if unrecognized > 0:
        print('Discarded {} games with no date'.format(unrecognized), file=sys.stderr)


def dump_since(since_time):
    """ Print all games generated since `since_time` from standard input """

    unrecognized = 0

    for line in sys.stdin:
        line = line.strip()
        d = sgf_to_time(line)

        if not d:
            unrecognized += 1
        elif d >= since_time:
            print(line)

    if unrecognized > 0:
        print('Discarded {} games with no date'.format(unrecognized), file=sys.stderr)


def main():
    """ Main function """

    if len(sys.argv) < 2:
        print('Usage: ./sgf2recent.py [number|date]')
        quit(1)

    since_time = str_to_time(sys.argv[1])

    if since_time:
        dump_since(since_time)
    else:
        number_of_games = int(sys.argv[1])

        dump_most_recent(number_of_games)


if __name__ == '__main__':
    main()
