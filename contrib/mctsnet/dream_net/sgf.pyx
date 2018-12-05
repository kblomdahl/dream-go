from .rules.board cimport Board
from .rules.color cimport BLACK, WHITE, opposite

import cython
from libc.stdlib cimport rand, RAND_MAX, malloc, free
from libc.time cimport time
cimport numpy as np
import numpy as np

cdef double random() nogil:
    return <double>rand() / (<double>RAND_MAX + 1.0)

@cython.boundscheck(False)
cdef int gather_moves(
    const unsigned char *line,
    const int line_length,
    int *color,
    int *x,
    int *y,
    int *winner
) nogil:
    cdef int i = 5
    cdef int resigned = 0
    cdef int count = 0

    winner[0] = 0

    # collect all played moves in the given SGF into pre-allocated arrays by
    # a stupid scanning algorithm that looks for the following patterns:
    #
    # - 'RE[...]'
    # - `;B[??]`
    # - `;W[??]`
    # - `;B[]`
    # - `;W[]`
    # - `Resign`
    #
    while i < line_length:
        if line[i] == 66 and line[i-1] == 91 and line[i-2] == 69 and line[i-3] == 82:  # RE[B
            winner[0] = BLACK
        elif line[i] == 87 and line[i-1] == 91 and line[i-2] == 69 and line[i-3] == 82:  # RE[W
            winner[0] = WHITE
        elif line[i] == 93 and line[i-3] == 91 and line[i-5] == 59:  # `;?[??]`
            if line[i-4] == 66 or line[i-4] == 87:  # `;B[??]` or `;W[??]`
                x[count] = line[i-2] - 97
                y[count] = line[i-1] - 97
                color[count] = BLACK if line[i-4] == 66 else WHITE
                count += 1
        elif line[i] == 93 and line[i-1] == 91 and line[i-3] == 59:  # `;?[]`
            if line[i-2] == 66 or line[i-2] == 87:  # `;B[]` or `;W[]`
                x[count] = 19
                y[count] = 19
                color[count] = BLACK if line[i-2] == 66 else WHITE
                count += 1
        elif line[i] == 100 and line[i-1] == 103 and line[i-2] == 105 and line[i-3] == 115 and line[i-4] == 101 and line[i-5] == 82:  # `Resign`
            resigned = 1

        i += 1

    # if this game was played to finish, then add potentially missing passing
    # moves at the end so that the engine can learn when it is appropriate to
    # pass
    if resigned == 0 and count > 2 and x[count-1] != 19 and y[count-1] != 19:
        x[count+0] = 19
        x[count+1] = 19
        y[count+0] = 19
        y[count+1] = 19

        color[count+0] = opposite(color[count-1])
        color[count+1] = color[count-1]
        count += 2

    return count

@cython.boundscheck(False)
cdef int _one(
    const unsigned char *line,
    const int line_length,
    Board board,
    int *winner,
    int *next1_color,
    int *next1_logits
) nogil:
    # gather the moves played without doing anything, this is done so that we
    # do not need to playout more of the game than we need to, and be able to
    # perform look-a-head during learning.
    #
    # It also helps since it allows us to discard games that are too short, and
    # the winner of said game is therefore uncertain.
    #
    cdef int *color = <int*>malloc(sizeof(int) * (line_length / 5))
    cdef int *x = <int*>malloc(sizeof(int) * (line_length / 5))
    cdef int *y = <int*>malloc(sizeof(int) * (line_length / 5))
    cdef int symmetry_index, total_moves, pluck_move, index, i

    try:
        total_moves = gather_moves(line, line_length, color, x, y, winner)

        # playout the game until the move that we are going to pluck, this is
        # necessary because we have to extract the features.
        pluck_move = <int>(total_moves * random())

        for i in range(pluck_move):
            if x[i] < 19 and y[i] < 19:  # not pass
                index = 19 * y[i] + x[i]

                if board._is_valid(color[i], index):
                    board._place(color[i], index)
                else:
                    return 0

        # return the tuple `(winner, color, logits)`, where `logits` is the
        # probability of each move being played, and `color` is the color of
        # the next move.
        next1_color[0] = color[pluck_move]

        if x[pluck_move] < 19:
            next1_logits[0] = 19 * y[pluck_move] + x[pluck_move]
        else:
            next1_logits[0] = 361

        return 1
    finally:
        free(color)
        free(x)
        free(y)

def one(line):
    cdef Board board = Board()

    # release the GIL (Global Interpreter Lock) while parsing the SGF file
    cdef unsigned char *line_ptr = <unsigned char*>line
    cdef int line_length = len(line)
    cdef int result, winner, next1_color, next1_logits

    with nogil:
        result = _one(line_ptr, line_length, board, &winner, &next1_color, &next1_logits)

    if result == 0:
        raise ValueError

    value = np.asarray([1.0 if winner == next1_color else -1.0], 'f2')
    policy = np.zeros((362,), 'f2')
    policy[next1_logits] = 1.0

    return (board, next1_color, value, policy)
