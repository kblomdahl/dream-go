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

LETTERS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]

def to_sgf_coord(x, y):
    return '{}{}'.format(LETTERS[x], LETTERS[y])

def to_sgf_heatmap(features, tower):
    AB = [] # black vertices
    AW = [] # white vertices
    LBs = []

    for y in range(19):
        for x in range(19):
            ch = to_sgf_coord(x, y)

            if features[3, y, x] > 0.0:
                AB += (ch,)
            elif features[4, y, x] > 0.0:
                AW += (ch,)

    num_channels = tower.shape[0]

    for i in range(num_channels):
        LB = {}

        for y in range(19):
            for x in range(19):
                ch = to_sgf_coord(x, y)

                if tower[i, y, x] > 1e-4:
                    LB[ch] = tower[i, y, x]

        LBs += (LB,)

    # x
    sgf = '{}{}'.format(
        'AB' + ''.join(map(lambda x: '[{}]'.format(x), AB)) if len(AB) > 0 else '',
        'AW' + ''.join(map(lambda x: '[{}]'.format(x), AW)) if len(AW) > 0 else '',
    )

    for LB in LBs:
        sgf += ';LB' + ''.join(map(lambda key: '[{}:{:.1f}]'.format(key[0], key[1]), LB.items())) if len(LB) > 0 else ''

    return sgf
