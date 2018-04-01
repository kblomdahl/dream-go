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
xxx
"""

import argparse
from datetime import datetime, timezone
import http.client
import re
import sys
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs


DT = re.compile(r'DT\[([^\]]*)\]')

def time_from_sgf(sgf):
    """ Retrieve the creation time of the given SGF """
    m = DT.search(sgf)

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

        return d.astimezone(timezone.utc) if d else datetime.utcnow()

    return datetime.utcnow()

parser = argparse.ArgumentParser(description='Upload data from standard input')
parser.add_argument('--sgf', action='store_true')
parser.add_argument('--bytes', type=int, nargs='?')
parser.add_argument('--date', type=int, nargs='?', help='Override the default date')
parser.add_argument('host', nargs=1)

args = parser.parse_args(sys.argv[1:])
if not args.sgf and not args.bytes:
    parser.print_help()
    quit()

# establish a connection to the provided host
url = urlparse(args.host[0])
query = parse_qs(url.query)
rest = http.client.HTTPConnection(url.netloc, timeout=60)

while not sys.stdin.closed:
    if args.sgf:
        payload = sys.stdin.readline()
        timestamp = time_from_sgf(payload)

        # convert to raw bytes so that it can be sent over HTTP
        payload = payload.encode('utf-8')
    elif args.bytes > 0:
        payload = sys.stdin.buffer.read(args.bytes)
        timestamp = datetime.utcnow()
    else:
        payload = sys.stdin.buffer.read()
        timestamp = datetime.utcnow()

    # quit if EOF
    if not payload:
        break

    # upload to the api
    query['date'] = '%d' % (args.date or timestamp.timestamp(),)

    rest.request(
        'POST',
        urlunparse((
            '',  # schema
            '',  # netloc
            url.path,  # path
            url.params,  # params
            urlencode(query, doseq=True),  # query
            url.fragment  # fragment
        )),
        payload
    )

    # check that we succeeded
    resp = rest.getresponse()
    resp.read()

    if resp.status != 200:
        print(payload)
