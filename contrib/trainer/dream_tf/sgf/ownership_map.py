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

from . import to_sgf_coord, to_sgf_initial_state


def to_sgf_ownership(features, ownership):
    """ Returns the SGF properties for a heat map of the given tensor """

    # collect the actual ownership
    ownership_labels = {}

    for y in range(19):
        for x in range(19):
            ch = to_sgf_coord(x, y)
            index = 19 * y + x

            if ownership[index] > 1e-4:
                ownership_labels[ch] = 'B'
            elif ownership[index] < 1e-4:
                ownership_labels[ch] = 'W'

    # pretty-print as a SGF properties (without the prefix)
    sgf = to_sgf_initial_state(features)

    if len(ownership_labels) > 0:
        sgf += ';LB' + ''.join(map(
            lambda key: '[{}:{}]'.format(key[0], key[1]), ownership_labels.items()
        ))

    return sgf
