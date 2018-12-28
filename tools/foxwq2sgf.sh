#!/bin/sh
# Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

INDEX_BASE_URL='http://happyapp.huanle.qq.com/cgi-bin/CommonMobileCGI/TXWQFetchChessList?type=2'
GAME_BASE_URL='http://happyapp.huanle.qq.com/cgi-bin/CommonMobileCGI/TXWQFetchChess?chessid='

# temporary files used to store the most recently downloaded index, or game
INDEX_FILE=$(mktemp)
GAME_FILE=$(mktemp)

trap "rm -f ${INDEX_FILE}" 0 1
trap "rm -f ${GAME_FILE}" 0 1

# loop over the entire index, until we encounter the end (indicated by an empty list of games)
index_url=${INDEX_BASE_URL}

while true ; do
    # download the current page of games
    curl -s "${index_url}" > ${INDEX_FILE}

    if test $(jq -c '.result' < ${INDEX_FILE}) -ne 0 ; then
        echo "Failed to download the index -- ${index_url}" >&2
        exit 1
    fi

    # download each game
    game_id_list=$(jq -rs '.[].chesslist[].chessid' < ${INDEX_FILE})

    if test -z "${game_id_list}" ; then
        exit 0
    fi

    for game_id in ${game_id_list} ; do
        output_file="data/foxwq/${game_id}.sgf"

        if test ! -f "${output_file}" ; then
            curl -s "${GAME_BASE_URL}${game_id}" > ${GAME_FILE} && \
                mkdir -p 'data/foxwq' && \
                jq -rs '.[].chess' < "${GAME_FILE}" > "${output_file}"
        fi

        echo "${game_id}"
    done

    # proceed to the next page
    last_game_id=$(echo "${game_id_list}" | tail -n 1)
    index_url="${INDEX_BASE_URL}&lastCode=${last_game_id}"
done
