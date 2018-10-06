// Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub trait SgfCoordinate {
    fn to_sgf(x: usize, y: usize) -> String;
}

pub struct CGoban;

impl SgfCoordinate for CGoban {
    fn to_sgf(x: usize, y: usize) -> String {
        const SGF_LETTERS: [char; 19] = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's'
        ];

        format!("{}{}", SGF_LETTERS[x], SGF_LETTERS[y])
    }
}

pub struct Sabaki;

impl SgfCoordinate for Sabaki {
    fn to_sgf(x: usize, y: usize) -> String {
        const SGF_LETTERS: [char; 19] = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's'
        ];

        format!("{}{}", SGF_LETTERS[x], SGF_LETTERS[18 - y])
    }
}