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

use std::ops::{Index, IndexMut};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Point {
    packed_index: u16
}

impl Point {
    pub const STRIDE: usize = 20;
    pub const MAX: usize = Self::STRIDE * 20 + 20;

    pub fn new(x: usize, y: usize) -> Self {
        debug_assert!(x < 19 && y < 19);

        let packed_index = Self::STRIDE * (y + 1) + (x + 1);

        Self {
            packed_index: packed_index as u16
        }
    }

    pub fn from_packed_parts(packed_index: usize) -> Self {
        if packed_index == 361 {
            Point::default()
        } else {
            let x = packed_index % 19;
            let y = packed_index / 19;

            Point::new(x, y)
        }
    }

    pub fn from_raw_parts(packed_index: u16) -> Self {
        debug_assert!(packed_index < Self::MAX as u16);

        Self { packed_index }
    }

    pub fn all() -> PointIter {
        PointIter::default()
    }

    pub fn x(&self) -> usize {
        const TO_X: [u8; Point::MAX] = [
            0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 0,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 17, 18, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 1, 2,
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 1, 2, 3, 4,
            5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 1,
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
        ];

        TO_X[self.to_i()] as usize
    }

    pub fn y(&self) -> usize {
        const TO_Y: [u8; Point::MAX] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
            10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
            17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18,
            18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
            19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
            19, 19, 19, 19, 19
        ];

        TO_Y[self.to_i()] as usize
    }

    pub fn offset(&self, dx: isize, dy: isize) -> Point {
        debug_assert!(-19 < dx && dx < 19);
        debug_assert!(-19 < dy && dy < 19);

        let delta = (Self::STRIDE as isize) * dy + dx;

        Point {
            packed_index:
                if delta < 0 {
                    self.packed_index.saturating_sub(-delta as u16)
                } else {
                    self.packed_index.saturating_add(delta as u16)
                }
        }
    }

    pub fn to_packed_index(&self) -> usize {
        if *self == Self::default() {
            361
        } else {
            19 * self.y() + self.x()
        }
    }

    pub(super) fn to_i(&self) -> usize {
        self.packed_index as usize
    }
}

impl ::std::fmt::Debug for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(f, "({}, {})", self.x(), self.y())
    }
}

impl Default for Point {
    fn default() -> Self {
        Self {
            packed_index: 0
        }
    }
}

macro_rules! define_index_type {
    ($type:ty) => {
        impl Index<Point> for [$type] {
            type Output = $type;

            fn index(&self, index: Point) -> &Self::Output {
                self.index(index.to_i())
            }
        }

        impl IndexMut<Point> for [$type] {
            fn index_mut(&mut self, index: Point) -> &mut Self::Output {
                self.index_mut(index.to_i())
            }
        }
    };
}

define_index_type!(u8);
define_index_type!(u16);
define_index_type!(u32);
define_index_type!(u64);
define_index_type!(usize);
define_index_type!(bool);
define_index_type!(Point);

pub struct PointIter {
    x: u8,
    y: u8,
}

impl Default for PointIter {
    fn default() -> Self {
        PointIter {
            x: 0,
            y: 0,
        }
    }
}

impl Iterator for PointIter {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {

        if self.x < 19 && self.y < 19 {
            let out = Point::new(self.x as usize, self.y as usize);

            self.x += 1;
            if self.x == 19 {
                self.x = 0;
                self.y += 1;
            }

            Some(out)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_set::HashSet;
    use super::*;

    #[test]
    fn identity() {
        let point = Point::new(3, 7);

        assert_eq!(point.x(), 3);
        assert_eq!(point.y(), 7);
    }

    #[test]
    fn offset_bottomleft() {
        let point = Point::new(0, 0);

        point.offset(-1, 0); // undefined, but should not panic
        point.offset(0, -1); // undefined, but should not panic
        assert_eq!(point.offset(1, 0), Point::new(1, 0));
        assert_eq!(point.offset(0, 1), Point::new(0, 1));
        assert_eq!(point.offset(1, 1), Point::new(1, 1));
    }

    #[test]
    fn offset_topright() {
        let point = Point::new(18, 18);

        point.offset(1, 0); // undefined, but should not panic
        point.offset(0, 1); // undefined, but should not panic
        assert_eq!(point.offset(-1,  0), Point::new(17, 18));
        assert_eq!(point.offset( 0, -1), Point::new(18, 17));
        assert_eq!(point.offset(-1, -1), Point::new(17, 17));
    }

    #[test]
    fn all_are_valid() {
        for point in Point::all() {
            assert_ne!(point, Point::default());
        }
    }

    #[test]
    fn non_is_default() {
        for point in Point::all() {
            assert_ne!(point, Point::default());
        }
    }

    #[test]
    fn less_than_max() {
        for point in Point::all() {
            assert!(point.to_i() < Point::MAX);
        }

        assert!(Point::default().to_i() < Point::MAX);
    }

    #[test]
    fn has_all_points() {
        assert_eq!(Point::all().collect::<HashSet<_>>().len(), 361);
    }
}
