// Copyright 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use color::Color;
use point::Point;

pub trait Vertex {
    fn empty() -> Self;
    fn invalid() -> Self;

    fn is_valid(self) -> bool;
    fn color(self) -> Option<Color>;
    fn next_point(self) -> Point;
    fn head_point(self) -> Point;
    fn num_liberties(self) -> usize;
    fn visited(self) -> bool;

    fn set_color(&mut self, color: Option<Color>);
    fn set_next_point(&mut self, next_point: Point);
    fn set_head_point(&mut self, head_point: Point);
    fn set_liberties(&mut self, num_liberties: usize);
    fn set_visited(&mut self, visited: bool);

    fn add_liberties(&mut self, delta: usize);
    fn sub_liberties(&mut self, delta: usize);
}

impl Vertex for u32 {
    fn empty() -> Self {
        0x0
    }

    fn invalid() -> Self {
        0x3
    }

    fn is_valid(self) -> bool {
        (self & 0x3) != 3
    }

    fn color(self) -> Option<Color> {
        const COLORS: [Option<Color>; 4] = [None, Some(Color::Black), Some(Color::White), None];

        COLORS[(self & 0x3) as usize]
    }

    fn next_point(self) -> Point {
        Point::from_raw_parts(((self & 0x00000ffc) >> 2) as u16)
    }

    fn head_point(self) -> Point {
        Point::from_raw_parts(((self & 0x003ff000) >> 12) as u16)
    }

    fn num_liberties(self) -> usize {
        ((self & 0x7fc00000) >> 22) as usize
    }

    fn visited(self) -> bool {
        (self & 0x80000000) != 0
    }

    fn set_color(&mut self, color: Option<Color>) {
        let value = match color {
            Some(Color::Black) => 1,
            Some(Color::White) => 2,
            _ => 0,
        };

        *self = (*self & 0xfffffffc) | (value as u32);
    }

    fn set_next_point(&mut self, next_vertex: Point) {
        *self = (*self & 0xfffff003) | (next_vertex.to_i() << 2) as u32;
    }

    fn set_head_point(&mut self, head_vertex: Point) {
        *self = (*self & 0xffc00fff) | (head_vertex.to_i() << 12) as u32;
    }

    fn set_liberties(&mut self, num_liberties: usize) {
        *self = (*self & 0x803fffff) | (num_liberties << 22) as u32;
    }

    fn set_visited(&mut self, visited: bool) {
        *self = (*self & 0x7fffffff) | if visited { 0x80000000 } else { 0 }
    }

    fn add_liberties(&mut self, delta: usize) {
        *self += (delta << 22) as u32;
    }

    fn sub_liberties(&mut self, delta: usize) {
        *self -= (delta << 22) as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_is_not_valid() {
        assert!(!u32::invalid().is_valid());
    }

    #[test]
    fn empty_is_valid() {
        assert!(u32::empty().is_valid());
    }

    #[test]
    fn exhaustive_vertex_bitfield() {
        let mut x: u32 = 0;

        for color in vec! [None, Some(Color::Black), Some(Color::White)].into_iter() {
            x.set_color(color);

            for next_vertex in Point::all() {
                x.set_next_point(next_vertex);
                x.set_head_point(next_vertex);

                for liberties in 0..32 {
                    x.set_liberties(liberties);

                    for visited in vec! [true, false] {
                        x.set_visited(visited);

                        assert!(x.is_valid());
                        assert_eq!(x.color(), color);
                        assert_eq!(x.next_point(), next_vertex);
                        assert_eq!(x.head_point(), next_vertex);
                        assert_eq!(x.num_liberties(), liberties);
                        assert_eq!(x.visited(), visited);
                    }
                }
            }
        }
    }

    #[test]
    fn change_liberty_count() {
        let mut x: u32 = 0;

        x.set_liberties(10);
        assert_eq!(x.num_liberties(), 10);

        x.add_liberties(2);
        assert_eq!(x.num_liberties(), 12);

        x.sub_liberties(4);
        assert_eq!(x.num_liberties(), 8);
    }
}
