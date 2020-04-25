// Copyright 2020 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use point::Point;

/// Iterator over all points that are directly adjacent to a given point. It
/// will yield one point for each cardinal direction:
/// 
/// * Right
/// * Down
/// * Left
/// * Up
pub struct AdjacentIter {
    starting_point: Point,
    position: usize,
}

impl AdjacentIter {
    pub fn new(starting_point: Point) -> Self {
        Self {
            starting_point: starting_point,
            position: 0,
        }
    }
}

impl Iterator for AdjacentIter {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        const DX: [i8; 4] = [1, 0, -1, 0];
        const DY: [i8; 4] = [0, -1, 0, 1];

        if self.position < 4 {
            let out = self.starting_point.offset(
                DX[self.position] as isize,
                DY[self.position] as isize
            );

            self.position += 1;
            Some(out)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_offset(point: Point, dx: isize, dy: isize) -> bool {
        let expected = point.offset(dx, dy);

        AdjacentIter::new(point).collect::<Vec<_>>().contains(&expected)
    }

    #[test]
    fn has_four() {
        let point = Point::new(9, 9);

        assert_eq!(AdjacentIter::new(point).count(), 4);
    }

    #[test]
    fn has_right() {
        assert!(has_offset(Point::new(9, 9), 1, 0));
    }

    #[test]
    fn has_down() {
        assert!(has_offset(Point::new(9, 9), 0, -1));
    }

    #[test]
    fn has_left() {
        assert!(has_offset(Point::new(9, 9), -1, 0));
    }

    #[test]
    fn has_up() {
        assert!(has_offset(Point::new(9, 9), 0, 1));
    }
}
