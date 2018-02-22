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
REST-ful server that speaks HTTP. It is used to organize all generated data,
and to query them in useful ways.

Uses SQLite3 as the internal storage engine.

Usage: ./database.py <database file>
"""

from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import sqlite3
import sys
from urllib.parse import parse_qs, urlparse


if len(sys.argv) < 2:
    print('Usage: ./server.py <database file>')
    quit()

def sqlite3_connection():
    """ Returns an SQLite3 connection to the database """
    return sqlite3.connect(sys.argv[1])

with sqlite3_connection() as conn:
    def create_table():
        """ Creat the master table if it does not already exist, there is also
        an implied column `rowid` that SQLite adds by itself that we use to
        identify games. """

        c = conn.cursor()

        # since we do not really need transactions turn a bunch of stuff off
        #
        # This is about 40x faster than the default settings...
        c.execute('PRAGMA synchronous = OFF')
        c.execute('PRAGMA journal_mode = MEMORY')

        # create the `collections` table
        c.execute('''
        CREATE TABLE IF NOT EXISTS collections (
            key VARCHAR(16) NOT NULL,
            timestamp DATE NOT NULL,
            generation BIGINT,
            data BLOB NOT NULL
        )
        ''')
        c.execute('''
        CREATE INDEX IF NOT EXISTS IX_collections ON collections (key, timestamp, generation)
        ''')
        conn.commit()
        c.close()

    create_table()

class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    pass

class RequestHandler(BaseHTTPRequestHandler):
    """
    The request handler, is current supports the `GET` and `POST` operations,
    where the path is used to control what path into the keystore to access.
    """

    def do_POST(self):
        """ Handle `POST` requests that add rows to the database. """

        url = urlparse(self.path)
        qs = parse_qs(url.query)

        # retrieve the key from the path
        key = url.path.strip('/')
        if '/' in key:
            self.send_response(400)  # bad request
            self.end_headers()

            return

        # retrieve the timestamp from the `date` parameter, or set it to the
        # current time if none is given.
        try:
            date = datetime.fromtimestamp(int(qs['date'][0]))
            if not date:
                date = datetime.utcnow()
        except (KeyError, TypeError):
            date = datetime.utcnow()

        # retrieve the generation from the `generation` parameter, or set it
        # to `None`.
        try:
            generation = int(qs['generation'][0])
        except (KeyError, ValueError):
            generation = None

        # read the content that the user sent us, we do this instead of
        # `self.rfile.read()` because that might read the next request if
        # the connection is re-used
        content_len = int(self.headers['Content-Length'] or 0)
        payload = self.rfile.read(content_len)

        # insert this item into the database
        with sqlite3_connection() as conn:
            c = conn.cursor()
            c.execute('''
            INSERT INTO collections (key, timestamp, generation, data)
                VALUES (?, ?, ?, ?)
            ''', (key, date, generation, payload))

            conn.commit()

        # acknowledgement that we received the `POST`
        self.send_response(200)
        self.end_headers()

    def do_EXACT(self, conn, key, rowid, field):
        """ Retrieve only the row that match the given `key` and
        `rowid`. """
        if field not in ['rowid', 'timestamp', 'generation', 'data']:
            self.send_response(400)  # bad request
            self.end_headers()

            return None

        c = conn.cursor()
        c.execute('''
        SELECT ''' + field + ''' FROM collections
            WHERE key = ? AND rowid = ?
        ''', (key, rowid))

        return c

    def do_RECENT(self, conn, key, n, field):
        """ Retrieve the `n` most recent entries for the given key
        `key`. """
        if field not in ['rowid', 'timestamp', 'generation', 'data']:
            self.send_response(400)  # bad request
            self.end_headers()

            return None

        c = conn.cursor()
        c.execute('''
        SELECT ''' + field + ''' FROM collections
            WHERE key = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (key, n))

        return c

    def do_GET(self):
        """ Handle `GET` requests """
        url = urlparse(self.path)

        try:
            key, query = url.path.strip('/').split('/', 1)
        except ValueError:
            self.send_response(400)  # bad request
            self.end_headers()
            return

        # parse the trailing part of the path as a query, it can take one
        # of the following forms:
        #
        # - [id]
        # - [id]/[field]
        # - recent/[n]
        # - recent/[n]/field
        parts = query.split('/')

        with sqlite3_connection() as conn:
            if len(parts) >= 1 and parts[0].isdigit():
                c = self.do_EXACT(
                    conn,
                    key,
                    int(parts[0]),
                    parts[1] if len(parts) > 1 else 'data'
                )
            elif len(parts) >= 2 and parts[0] == 'recent' and parts[1].isdigit():
                c = self.do_RECENT(
                    conn,
                    key,
                    int(parts[1]),
                    parts[2] if len(parts) > 2 else 'data'
                )
            else:
                self.send_response(400)  # bad request
                self.end_headers()

                return

            if c:
                # acknowledgement that we received the `GET` and send all of
                # the responses
                self.send_response(200)
                self.end_headers()

                while True:
                    row = c.fetchone()
                    if not row:
                        break

                    try:
                        self.wfile.write(row[0])
                    except TypeError:
                        self.wfile.write(str(row[0]).encode('utf-8'))

# spin up the http server
server = ThreadingSimpleServer(('', 8080), RequestHandler)
server.serve_forever()
