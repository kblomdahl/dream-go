#!/usr/bin/env python3

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

include_dirs = ['.', np.get_include()]
extensions = [
    Extension('dream_net.rules.board', ['dream_net/rules/board.pyx'], include_dirs=include_dirs),
    Extension('dream_net.rules.color', ['dream_net/rules/color.pyx'], include_dirs=include_dirs),
    Extension('dream_net.sgf', ['dream_net/sgf.pyx'], include_dirs=include_dirs),
]

setup(
    name='dream_net',
    packages=['dream_net', 'dream_net.rules'],
    ext_modules=cythonize(extensions)
)
