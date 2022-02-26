// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use dg_go::{Color, Point};

use std::str::FromStr;

#[derive(Debug, PartialEq)]
pub enum SgfToken<'a> {
    Add { color: &'a [u8], point: &'a [u8] },
    Handicap { text: &'a [u8] },
    Komi { text: &'a [u8] },
    Play { color: &'a [u8], point: &'a [u8] },
    Result { text: &'a [u8] },
    Size { text: &'a [u8] },
    Territory { color: &'a [u8], point: &'a [u8] },
}

impl<'a> SgfToken<'a> {
    pub fn point(&self) -> Point {
        match *self {
            Self::Add { color: _, point } => point_from_bytes(point),
            Self::Play { color: _, point } => point_from_bytes(point),
            Self::Territory { color: _, point } => point_from_bytes(point),
            _ => unreachable!()
        }
    }

    pub fn color(&self) -> Color {
        match *self {
            Self::Add { color, point: _ } if color == b"B" => Color::Black,
            Self::Add { color, point: _ } if color == b"W" => Color::White,
            Self::Play { color, point: _ } if color == b"B" => Color::Black,
            Self::Play { color, point: _ } if color == b"W" => Color::White,
            Self::Territory { color, point: _ } if color == b"B" => Color::Black,
            Self::Territory { color, point: _ } if color == b"W" => Color::White,
            _ => unreachable!()
        }
    }

    pub fn number(&self) -> f32 {
        match *self {
            Self::Handicap { text } => number_from_bytes(text),
            Self::Komi { text } => number_from_bytes(text),
            Self::Size { text } => number_from_bytes(text),
            _ => unreachable!()
        }
    }
}

fn point_from_bytes(b: &[u8]) -> Point {
    if b.len() != 2 {
        Point::default()
    } else {
        let x = b[0] - b'a';
        let y = b[1] - b'a';

        Point::new(x as usize, y as usize)
    }
}

fn number_from_bytes(b: &[u8]) -> f32 {
    f32::from_str(String::from_utf8_lossy(b).as_ref()).unwrap_or(f32::NAN)
}
