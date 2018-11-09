#!/usr/bin/env python3

from subprocess import Popen, PIPE, DEVNULL
from os import getcwd
import itertools
import re

def sgf_to_gtp(color, vertex):
    LETTERS = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'
    ]

    x = ord(vertex[0]) - 97
    y = ord(vertex[1]) - 97

    return 'play {} {}{}'.format(color, LETTERS[x], 19 - y)

class FinalScore:
    def __init__(self, score):
        self.score = score

    def expect_to(self, check):
        color = self.score[0]
        score = self.score[2:]
        result = check(color, score)

        if result is not None:
            raise ValueError(result)

class GenMove:
    def __init__(self, color, vertex):
        self.color = color
        self.vertex = vertex

    def expect_to(self, check):
        result = check(self.color, self.vertex)

        if result is not None:
            raise ValueError(result)

class TestCase:
    def __init__(self, name, num_rollout = 1600):
        self.name = name
        self.num_rollout = num_rollout

    def __enter__(self):
        self.proc = Popen([
            getcwd() + '/dream_go',
            '--noponder',
            '--num-rollout', str(self.num_rollout)
        ], bufsize=0, encoding='utf-8', stdin=PIPE, stdout=PIPE, stderr=DEVNULL)

        stdin = self.proc.stdin
        stdin.write('boardsize 19\n')
        stdin.write('komi 7.5\n')
        stdin.write('clear_board\n')

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        try:
            self.proc.communicate()
        except:
            self.proc.kill()

        if exception_value is not None:
            print('{}... failed ({})'.format(self.name, exception_value))
        else:
            print('{}... ok'.format(self.name))

        return isinstance(exception_value, ValueError)

    def setup_from_sgf(self, file_name, count):
        MOVE = re.compile(r';([BW])\[([a-s]{2})\]')
        stdin = self.proc.stdin

        with open(file_name, 'r') as file:
            content = file.read()

            for match in itertools.islice(re.finditer(MOVE, content), count or 722):
                stdin.write(sgf_to_gtp(match.group(1), match.group(2)) + '\n')
                stdin.flush()

    def final_score(self):
        stdin = self.proc.stdin
        stdin.write('1000 final_score\n')
        stdin.flush()

        for line in self.proc.stdout:
            if line.startswith('=1000'):
                parts = line.split(' ')

                return FinalScore(parts[1].strip())

    def genmove(self, color):
        stdin = self.proc.stdin
        stdin.write('1000 genmove {}\n'.format(color))
        stdin.flush()

        for line in self.proc.stdout:
            if line.startswith('=1000'):
                parts = line.split(' ')

                return GenMove(color, parts[1].strip())

def be_vertex(color, vertex):
    def _be_vertex(other_color, other_vertex):
        if other_color != color:
            return 'expected color {}, got {}'.format(color, other_color)
        if other_vertex != vertex:
            return 'expected vertex {}, got {}'.format(vertex, other_vertex)

    return _be_vertex

def not_be_vertex(color, vertex):
    def _not_be_vertex(other_color, other_vertex):
        if other_color != color:
            return 'expected color {}, got {}'.format(color, other_color)
        if other_vertex == vertex:
            return 'expected not vertex {}, got {}'.format(vertex, other_vertex)

    return _not_be_vertex

def be_winner(color, score):
    def _be_winner(other_color, other_score):
        if color is not None and color != other_color:
            return 'expected color {} to win, got {}+{}'.format(color, other_color, other_score)
        if score is not None and score != other_score:
            return 'expected score {}, got {}'.format(score, other_score)

    return _be_winner

