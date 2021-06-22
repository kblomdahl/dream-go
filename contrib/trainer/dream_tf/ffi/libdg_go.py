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

from cffi import FFI

def load_shared_library(ffi):
    for library_name in ['./libdg_go.so', './dg_go.dll']:
        try:
            return ffi.dlopen(library_name)
        except:
            pass

    print('Failed to load the shared library -- libdg_go.so')
    quit(1)


# -------- Simple FFI (independent functions) --------


SIMPLE_FFI = FFI()
SIMPLE_FFI.cdef("""
    int get_num_features();
    void set_seed(int seed);
""")

SIMPLE_LIB = load_shared_library(SIMPLE_FFI)


def get_num_features():
    """ Returns the number of features that the Go engine will use. """
    return SIMPLE_LIB.get_num_features()


def set_seed(seed):
    """ Sets the seed of the extraction """
    return SIMPLE_LIB.set_seed(seed)
