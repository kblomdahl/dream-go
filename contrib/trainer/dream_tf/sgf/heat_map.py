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

from . import to_sgf_coord


def to_sgf_heat_map(features, tower):
    """ Returns the SGF properties for a heat map of the given tensor """

    black_vertices = []
    white_vertices = []

    for y in range(19):
        for x in range(19):
            ch = to_sgf_coord(x, y)

            if features[3, y, x] > 0.0:
                black_vertices += (ch,)
            elif features[4, y, x] > 0.0:
                white_vertices += (ch,)

    # collect the actual heat map
    num_channels = tower.shape[0]
    labels = []

    for i in range(num_channels):
        labels_for_channel = {}

        for y in range(19):
            for x in range(19):
                ch = to_sgf_coord(x, y)

                if tower[i, y, x] > 1e-4:
                    labels_for_channel[ch] = tower[i, y, x]

        labels += (labels_for_channel,)

    # pretty-print as a SGF properties (without the prefix)
    sgf = '{}{}'.format(
        'AB' + ''.join(map(lambda v: '[{}]'.format(v), black_vertices)) if len(black_vertices) > 0 else '',
        'AW' + ''.join(map(lambda v: '[{}]'.format(v), white_vertices)) if len(white_vertices) > 0 else '',
    )

    for labels_for_channel in labels:
        if len(labels_for_channel) > 0:
            sgf += ';LB' + ''.join(map(
                lambda key: '[{}:{:.1f}]'.format(key[0], key[1]), labels_for_channel.items()
            ))

    return sgf
