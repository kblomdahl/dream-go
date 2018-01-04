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

Usage: ./sgf2recent.py [number] < kgs_big.sgf > kgs_big_recent.sgf
"""
from datetime import datetime
import heapq
import sys
import re

def main(k):
    """ Main function """

    DT = re.compile(r'DT\[([^\]]*)\]')

    # collect all lines into a min-heap, discarding the minimum
    # element everytime it grows too large
    entries = []
    unrecognized = 0
    count = 0

    for line in sys.stdin:
        line = line.strip()
        m = DT.search(line)

        if m:
            d = None

            try:
                # check for our extended date-time format first (which
                # includes the time with a higher precision)
                if not d:
                    d = datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%S%z')
            except ValueError:
                pass

            try:
                # this is the Tygem SGF format
                if not d:
                    d = datetime.strptime(m.group(1), '%Y.%m.%d %H:%M')
            except ValueError:
                pass

            try:
                # this is the standard SGF format
                if not d:
                    d = datetime.strptime(m.group(1), '%Y-%m-%d')
            except ValueError:
                pass

            if not d:
                unrecognized += 1
            else:
                item = (d, line)

                if len(entries) > k:
                    heapq.heapreplace(entries, item)
                else:
                    heapq.heappush(entries, item)
        else:
            unrecognized += 1

        count += 1

    # print each entry, from the oldest to the most recent:
    if len(entries) > k:
        heapq.heappop(entries)  # discard the buffer element

    for (_, line) in [heapq.heappop(entries) for _ in range(len(entries))]:
        print(line)

    if unrecognized > 0:
        print('Discarded {} games no date'.format(unrecognized), file=sys.stderr)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: ./sgf2recent.py [number]')
        quit()

    main(int(sys.argv[1]))
