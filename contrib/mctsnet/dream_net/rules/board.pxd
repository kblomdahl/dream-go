cimport numpy as np

cdef class Board:
    # -------- Instance variables --------

    cdef char vertices[368]
    cdef int next_vertex[368]
    cdef unsigned long zobrist_hash

    # Set of the most recent board positions, used to detect super-ko. This can
    # miss some very long ko's but those should not occur in real games anyway.
    cdef unsigned long zobrist_hashes[8]
    cdef int zobrist_hashes_index

    # -------- Methods --------

    cpdef Board copy(self)

    cdef int _has_one_liberty(self, int index) nogil
    cdef int _has_two_liberty(self, int index) nogil

    cdef unsigned long _capture_ko(self, int index) nogil

    cdef int _is_valid(self, int color, int index) nogil
    cdef int _is_ko(self, int color, int index) nogil
    cpdef int is_valid(self, int color, int x, int y)
    cpdef int is_valid_aux(self, int color, int index)

    cdef void _capture(self, int index) nogil
    cdef void _connect_with(self, int index, int other) nogil
    cdef void _place(self, int color, int index) nogil
    cpdef void place(self, int color, int x, int y)
    cpdef void place_aux(self, int color, int index)

    cdef int _get_pattern_code(self, int color, int index) nogil
    cdef int _get_pattern(self, int color, int index) nogil
    cpdef int get_pattern(self, int color, int x, int y)

    cdef int _get_num_liberties(self, int index) nogil
