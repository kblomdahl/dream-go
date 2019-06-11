# Copyright (c) 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

"""
Foreign function interface for the functions defined in `src/libdg_go/utils/extract_example.rs`.
"""

from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
cimport numpy as np

ctypedef struct Example:
    int index
    int index_next
    int color
    float[362] policy
    float[362] policy_next
    float[361] ownership
    int winner
    int number
    float komi
    float[0] features

cpdef extern int get_num_features() nogil
cdef extern int extract_single_example(const unsigned char* raw_sgf_content, Example* out) nogil


cpdef parse_single_example(const unsigned char* line):
    cdef int result
    cdef int num_features = get_num_features()
    cdef int features_size = 361 * num_features
    cdef Example* example = <Example*>PyMem_Malloc(sizeof(Example) + sizeof(float) * features_size)
    if not example:
        raise MemoryError()

    try:
        with nogil:
            result = extract_single_example(line, example)
        if result != 0:
            raise ValueError('Could not extract example from line -- ' + str(result))

        ownership = np.asarray(<np.float32_t[:361]> example.ownership)
        policy = np.array(<np.float32_t[:362]> example.policy)
        policy_next = np.array(<np.float32_t[:362]> example.policy_next)
        score = np.zeros([723], 'f4')
        ownership = np.stack([
            ownership > 0.0,
            ownership < 0.0
        ]).reshape([19, 19, 2]).astype('f4')

        # fix any partial policies
        policy[example.index] += 1.0 - np.sum(policy)
        policy_next[example.index_next] += 1.0 - np.sum(policy_next)

        # set the score
        actual_score = 361 + np.sum(ownership > 0.0) - np.sum(ownership < 0.0)
        score[actual_score] = 1.0

        # fix any missing ownerships to `(50%, 50%)`
        ownership += np.asarray([0.5, 0.5], 'f4') * (1.0 - np.sum(ownership, axis=-1, keepdims=True))

        return (
            np.array(<np.float32_t[:features_size]> example.features).reshape([19, 19, num_features]),
            {
                'policy': policy,
                'policy_next': policy_next,
                'value': np.asarray([1.0, 0.0] if example.winner == example.color else [0.0, 1.0], 'f4'),
                'score': score,
                'ownership': ownership
            }
        )
    finally:
        PyMem_Free(example)
