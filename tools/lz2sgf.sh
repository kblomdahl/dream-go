#!/bin/sh
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

for file in "$@" ; do
    running_procs=0

    zipinfo -1 "${file}" | while read archive ; do
        (unzip -p "${file}" "${archive}" | gunzip | ./tools/lz_decode2sgf.py) &

        running_procs=$((${running_procs} + 1))
        if test ${running_procs} -eq 16 ; then
            running_procs=0
            wait
        fi
    done

    wait
done
