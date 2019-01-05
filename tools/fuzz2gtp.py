#!/usr/bin/env python3
# Copyright 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0103, C0301

"""
Super simple fuzzing tool for GTP clients, and generate _self-play_ games forever
using the GTP interface.

Usage: ./fuzz2gtp.py | ./dream_go --gtp
"""

count = 0

while True:
    if count == 0:
        print('clear_board')
    count = (count + 1) % 150

    print('genmove b')
    print('genmove w')

