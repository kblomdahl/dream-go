// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use dg_go::Point;

use std::fmt;
use std::str;

/// The letters used in the GTP protocol to represent the Y coordinate.
/// 
/// These letters are excluding `i` as indicated by the specification section
/// 2.11.
const LETTERS: [char; 25] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z'
];

/// The coordinates of a crossing on the go board.
#[derive(Debug, PartialEq)]
pub struct Vertex {
    pub x: usize,
    pub y: usize
}

impl Vertex {
    /// Returns if this is a passing move.
    pub fn is_pass(&self) -> bool {
        self.x >= 19 || self.y >= 19
    }
}

impl From<Point> for Vertex {
    fn from(point: Point) -> Self {
        let x = point.x();
        let y = point.y();

        Self { x, y }
    }
}

impl str::FromStr for Vertex {
    type Err = ();

    fn from_str(s: &str) -> Result<Vertex, Self::Err> {
        let s = s.to_lowercase();

        if s == "pass" {
            Ok(Vertex {x: 19, y: 19})
        } else if s.len() < 2 {
            Err(())
        } else {
            let mut chars = s.chars();
            let x = LETTERS.binary_search(&chars.next().unwrap());
            let y = chars.collect::<String>().parse::<usize>();

            if let (Ok(x), Ok(y)) = (x, y) {
                Ok(Vertex {x: x, y: y - 1})
            } else {
                Err(())
            }
        }
    }
}

impl fmt::Display for Vertex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", LETTERS[self.x], self.y + 1)
    }
}
