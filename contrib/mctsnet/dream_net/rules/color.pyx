BLACK = 1  # Constant used to identify the BLACK color
WHITE = 2  # Constant used to identify the WHITE color

cpdef int opposite(int color) nogil:
    """ Returns the opposite of the given color. """
    if color == 1:
        return 2
    elif color == 2:
        return 1
    else:
        return color
