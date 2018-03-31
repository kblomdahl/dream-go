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

from datetime import datetime
import dateutil.parser
from http.server import HTTPServer, BaseHTTPRequestHandler
import petname
from socketserver import ThreadingMixIn
import sys
import sqlite3
import threading
from urllib.parse import urlsplit, parse_qs, unquote

def sqlite3_execute(callback):
    """ Calls the given function with an SQLite3 cursor """

    with sqlite3.connect(sys.argv[1]) as conn:
        cursor = conn.cursor()
        cursor.execute('PRAGMA synchronous = OFF')
        cursor.execute('PRAGMA journal_mode = MEMORY')

        try:
            callback(cursor)
        finally:
            conn.commit()

def ensure_table_exists(cursor, table):
    """ Create the given table in the SQL database if not does not already
    exist. """

    if not all([c.isalnum() or c == '_' for c in table]):
        raise ValueError('invalid table name')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS %s (
        name TEXT NOT NULL,
        parent TEXT,
        creationtime DATETIME NOT NULL,
        content BLOB NOT NULL
    )
    ''' % table)

    cursor.execute('''
    CREATE INDEX IF NOT EXISTS %s ON %s (
        creationtime DESC,
        parent
    )
    ''' % ('ix_' + table, table))


class DBRequest:
    """ """

    operation = None
    table = None
    parent = None
    creationtime = None
    separator = None

    def __init__(self, path):
        url = urlsplit(path)

        # determine the operation and target from the path:
        #
        # - `[table]/recent/[n]`
        # - `[table]/recent/[n]/[field]`
        # - `[table]/count/[field]`
        # - `[table]/[name]`
        # - `[table]`
        #
        parts = list(filter(None, unquote(url.path).split('/')))
        if not parts:
            raise ValueError('missing table name')

        self.table = parts[0]

        if len(parts) == 2:
            self.operation = ('SELECT', parts[1])
        elif len(parts) >= 3 and parts[1] == 'recent':
            try:
                n = int(parts[2])

                if len(parts) >= 4:
                    assert parts[3] in ['rowid', 'name', 'parent', 'creationtime']

                    self.operation = ('RECENT', n, parts[3])
                else:
                    self.operation = ('RECENT', n, 'content')
            except ValueError:
                raise ValueError('unrecognized request')
        elif len(parts) == 3 and parts[1] == 'count':
            assert parts[2] in ['rowid', 'name', 'parent', 'creationtime']

            self.operation = ('COUNT', parts[2])
        else:
            raise ValueError('unrecognized request')

        # set any optional parameters from the query string:
        #
        # - creationtime
        # - parent
        # - separator
        #
        if url.query:
            query = parse_qs(url.query, strict_parsing=True)

            try:
                self.parent = query['parent'][0]
            except KeyError:
                pass

            try:
                creationtime = int(query['creationtime'][0])

                self.creationtime = datetime.fromtimestamp(creationtime)
            except ValueError:
                raise ValueError('invalid creation time')
            except KeyError:
                pass

            try:
                self.separator = query['separator'][0]
            except KeyError:
                pass

class DBRequestHandler(ThreadingMixIn, BaseHTTPRequestHandler):
    def do_POST(self):
        """ Handle POST requests that adds records to the database """

        request = DBRequest(self.path)

        if request:
            def _add_entry(cursor):
                if not request.creationtime:
                    request.creationtime = datetime.now()

                name = petname.Generate(4, '_')

                # read the content that the user sent us, we do this instead of
                # `self.rfile.read()` because that might read the next request if
                # the connection is re-used
                content_len = int(self.headers['Content-Length'] or 0)
                content = self.rfile.read(content_len)

                # check that the table we're inserting into exists (and validate
                # the table name), and then insert the new row into the table.
                ensure_table_exists(cursor, request.table)

                cursor.execute('''
                INSERT INTO %s (name, parent, creationtime, content)
                    VALUES (?, ?, ?, ?)
                ''' % (request.table,), (name, request.parent, request.creationtime, content)
                )

            sqlite3_execute(_add_entry)

            # acknowledge the request from the client when we are done
            self.send_response(200)
            self.end_headers()

    def _write_results(self, cursor, request):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        first = True

        while True:
            row = cursor.fetchone()
            if not row:
                break

            if not first and request.separator:
                self.wfile.write(request.separator.encode('utf-8'))
            first = False

            for (i, value) in enumerate(row):
                if isinstance(value, int):
                    value = '%d' % (value,)
                if isinstance(value, datetime):
                    value = ''

                try:
                    if i > 0:
                        self.wfile.write('|'.encode('utf-8'))
                    self.wfile.write(value)
                except TypeError:
                    self.wfile.write(value.encode('utf-8'))

        self.wfile.flush()

    def do_SELECT(self, cursor, request):
        ensure_table_exists(cursor, request.table)
        cursor.execute('''
        SELECT content FROM %s WHERE name = ?
        ''' % (request.table), (request.operation[1],))

        self._write_results(cursor, request)

    def do_RECENT(self, cursor, request):
        ensure_table_exists(cursor, request.table)
        cursor.execute('''
        SELECT %s FROM %s ORDER BY creationtime DESC LIMIT ?
        ''' % (request.operation[2], request.table), (request.operation[1],))

        self._write_results(cursor, request)

    def do_COUNT(self, cursor, request):
        ensure_table_exists(cursor, request.table)
        cursor.execute('''
        SELECT %s, COUNT(*) FROM %s GROUP BY %s
        ''' % (request.operation[1], request.table, request.operation[1]))

        self._write_results(cursor, request)

    def do_GET(self):
        """ Handle GET requests that retrieve data from the database """

        try:
            request = DBRequest(self.path)

            if not request.operation:
                raise ValueError('missing query')
            elif request.operation[0] == 'SELECT':
                sqlite3_execute(lambda cursor: self.do_SELECT(cursor, request))
            elif request.operation[0] == 'RECENT':
                sqlite3_execute(lambda cursor: self.do_RECENT(cursor, request))
            elif request.operation[0] == 'COUNT':
                sqlite3_execute(lambda cursor: self.do_COUNT(cursor, request))
            else:
                raise ValueError('unrecognized request')
        except ValueError as e:
            self.send_response(400)
            self.end_headers()

            raise e

if __name__ == '__main__':
    if not sys.argv[1:]:
        print('Usage: ./dream_db [database]')
        quit()

    httpd = HTTPServer(('', 8080), DBRequestHandler)
    httpd.serve_forever()
