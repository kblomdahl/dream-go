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

use std::fmt;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Color {
    Black = 1,
    White = 2
}

impl Color {
    /// Returns the opposite of this color.
    pub fn opposite(&self) -> Color {
        match *self {
            Color::Black => Color::White,
            Color::White => Color::Black
        }
    }
}

impl ::std::str::FromStr for Color {
    type Err = ();

    fn from_str(s: &str) -> Result<Color, Self::Err> {
        let s = s.to_lowercase();

        if s == "black" || s == "b" {
            Ok(Color::Black)
        } else if s == "white" || s == "w" {
            Ok(Color::White)
        } else {
            Err(())
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Color::Black => write!(f, "B"),
            Color::White => write!(f, "W")
        }
    }
}
