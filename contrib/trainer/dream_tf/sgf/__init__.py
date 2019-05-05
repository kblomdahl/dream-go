# Copyright (c) 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

LETTERS = (
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
)


def to_sgf_coord(x, y):
    return '{}{}'.format(LETTERS[x], LETTERS[y])

def to_sgf_initial_state(features):
    black_vertices = []
    white_vertices = []

    for y in range(19):
        for x in range(19):
            ch = to_sgf_coord(x, y)

            if features[y, x, 5] > 0.0:  # player
                black_vertices += (ch,)
            elif features[y, x, 21] > 0.0:  # opponent
                white_vertices += (ch,)

    # pretty-print as a SGF properties (without the prefix)
    return '{}{}'.format(
        'AB' + ''.join(map(lambda v: '[{}]'.format(v), black_vertices)) if len(black_vertices) > 0 else '',
        'AW' + ''.join(map(lambda v: '[{}]'.format(v), white_vertices)) if len(white_vertices) > 0 else '',
    )
