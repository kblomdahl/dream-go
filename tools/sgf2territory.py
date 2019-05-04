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
Reads a _big_ SGF file from standard input and writes to standard output the
same games after the territory of the final board state has been determined.

Takes as input the `CUDA_VISIBLE_DEVICES` to set for each engine.

Usage: ./sgf2territory.py 0 0 1 1 < kgs_big.sgf > kgs_territory.sgf
"""
from subprocess import Popen, PIPE, DEVNULL, TimeoutExpired
from threading import Thread, Lock
from tempfile import NamedTemporaryFile
from queue import Queue

import re
import sys

RIGHT_PAREN = re.compile(r'\)$')
PRINT_LOCK = Lock()
GTP_TO_SGF = {
    'a': 'a',
    'b': 'b',
    'c': 'c',
    'd': 'd',
    'e': 'e',
    'f': 'f',
    'g': 'g',
    'h': 'h',
    'j': 'i',
    'k': 'j',
    'l': 'k',
    'm': 'l',
    'n': 'm',
    'o': 'n',
    'p': 'o',
    'q': 'p',
    'r': 'q',
    's': 'r',
    't': 's',
}

def gtp_to_sgf(coord):
    x = GTP_TO_SGF[coord[0]]
    y = 18 - (int(coord[1:]) - 1)

    return x + chr(ord('a') + y)

def inf_to_territory(line, line_nr):
    count = 0

    for value in line.split(' '):
        if not value:
            continue

        coordinate = chr(ord('a') + count) + chr(ord('a') + line_nr)
        value = float(value)

        yield (coordinate, value)
        count += 1

def score_game(engine, sgf):
    """
    Returns the territory of the game in the given SGF file as
    judged by `dream_go`.
    """

    with NamedTemporaryFile() as sgf_file:
        sgf_file.write(sgf.encode())
        sgf_file.flush()

        engine.stdin.write('1000 loadsgf {}\n'.format(sgf_file.name))
        engine.stdin.write('2000 initial_influence B territory_value\n')
        engine.stdin.write('3000 list_stones black\n')
        engine.stdin.write('4000 list_stones white\n')
        engine.stdin.write('5000 clear_board\n')
        engine.stdin.flush()

        black_territory = set()
        white_territory = set()

        for line in engine.stdout:
            line = line.strip().lower()

            if line.startswith('?'):
                break
            elif line.startswith('=2000'):
                line = line[5:].strip()
                line_nr = 0
                black_territory |= set([coord for (coord, value) in inf_to_territory(line, line_nr) if value < 0.0])
                white_territory |= set([coord for (coord, value) in inf_to_territory(line, line_nr) if value > 0.0])

                for line in engine.stdout:
                    line = line.strip().lower()
                    if not line:
                        break

                    line_nr += 1
                    black_territory |= set([coord for (coord, value) in inf_to_territory(line, line_nr) if value < 0.0])
                    white_territory |= set([coord for (coord, value) in inf_to_territory(line, line_nr) if value > 0.0])
            elif line.startswith('=3000'):
                black_stones = set(map(gtp_to_sgf, line.split(' ')[1:]))
                black_territory |= {b for b in black_stones if b not in white_territory}
            elif line.startswith('=4000'):
                white_stones = set(map(gtp_to_sgf, line.split(' ')[1:]))
                white_territory |= {w for w in white_stones if w not in black_territory}
            elif line.startswith('=5000'):
                break

        return black_territory, white_territory


class ScoreThread(Thread):
    def __init__(self, queue, visible_devices):
        Thread.__init__(self, daemon=False)

        self.queue = queue
        self.visible_devices = visible_devices
        self.engine = None
        self.count = 0
        self.start()


    def start_engine(self):
        engine = Popen(
            [
                '/usr/games/gnugo',
                '--mode', 'gtp',
                '--score', 'aftermath',
                '--play-out-aftermath',
                '--chinese-rules',
                '--cache-size', '512M'
            ],
            stdin=PIPE,
            stdout=PIPE,
            stderr=DEVNULL,
            encoding='UTF-8',
            env={'CUDA_VISIBLE_DEVICES': self.visible_devices}
        )

        # start a background thread that will kill this engine after three minutes
        def _kill_engine():
            try:
                engine.wait(180)
            except TimeoutExpired:
                engine.kill()

        Thread(target=_kill_engine, daemon=True).start()

        return engine

    def get_or_create_engine(self):
        if self.engine and self.count % 100 == 0:
            self.engine.communicate('quit\n')
            self.engine = None

        if not self.engine:
            with PRINT_LOCK:
                print(
                    '[{}] Restarting the scoring engine after {} games'.format(self.visible_devices, self.count),
                    file=sys.stderr,
                    flush=True
                )
            self.engine = self.start_engine()

        self.count += 1
        return self.engine


    def try_score_game(self, line):
        while True:
            try:
                return score_game(self.get_or_create_engine(), line)
            except BrokenPipeError:
                self.engine = None


    def run(self):
        try:
            while True:
                line = self.queue.get()
                if not line:
                    break

                black, white = self.try_score_game(line)

                if black:
                    line = re.sub(RIGHT_PAREN, 'TB' + ''.join(map(lambda x: '[{}]'.format(x), black)) + ')', line)
                if white:
                    line = re.sub(RIGHT_PAREN, 'TW' + ''.join(map(lambda x: '[{}]'.format(x), white)) + ')', line)

                with PRINT_LOCK:
                    print(line, flush=True)

                self.queue.task_done()
        finally:
            if self.engine:
                self.engine.communicate('quit\n')
                self.engine = None


def main():
    """ Main function """
    pending_scoring = Queue(maxsize=100)
    threads = list([ScoreThread(pending_scoring, visible_devices) for visible_devices in sys.argv[1:]])

    for line in sys.stdin:
        line = line.strip()

        pending_scoring.put(line)
    pending_scoring.join()

    for _ in threads:
        pending_scoring.put(None)
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
