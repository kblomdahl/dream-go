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

static SGF_LETTERS: [char; 26] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
];

pub trait SgfCoordinate {
    fn to_sgf(x: usize, y: usize) -> String;
    fn parse(s: &str) -> Result<(usize, usize), SgfCoordinateError>;
}

pub enum SgfCoordinateError {
    InvalidLen,
    UnrecognizedCharacter
}

pub struct CGoban;

impl SgfCoordinate for CGoban {
    fn to_sgf(x: usize, y: usize) -> String {
        format!("{}{}", SGF_LETTERS[x], SGF_LETTERS[y])
    }

    fn parse(s: &str) -> Result<(usize, usize), SgfCoordinateError> {
        if s.len() == 0 {
            Ok((19, 19))
        } else if s.len() == 2 {
            let mut ch = s.chars();
            let x = ch.next().and_then(|x| { SGF_LETTERS.binary_search(&x).ok() });
            let y = ch.next().and_then(|y| { SGF_LETTERS.binary_search(&y).ok() });

            match (x, y) {
                (Some(x), Some(y)) => {
                    if x >= 19 || y >= 19 {
                        Ok((19, 19))
                    } else {
                        Ok((x, y))
                    }
                },
                _ => Err(SgfCoordinateError::UnrecognizedCharacter)
            }
        } else {
            Err(SgfCoordinateError::InvalidLen)
        }
    }
}

pub struct Sabaki;

impl SgfCoordinate for Sabaki {
    fn to_sgf(x: usize, y: usize) -> String {
        format!("{}{}", SGF_LETTERS[x], SGF_LETTERS[18 - y])
    }

    fn parse(s: &str) -> Result<(usize, usize), SgfCoordinateError> {
        match CGoban::parse(s) {
            Ok((x, y)) if x < 19 && y < 19 => {
                Ok((x, 18 - y))
            },
            err => err
        }
    }
}
