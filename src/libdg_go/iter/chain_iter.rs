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

pub trait NextLink {
    fn next_link(&self, point: Point) -> Point;
}

///
pub struct ChainIter<L: NextLink> {
    starting_point: Point,
    previous_point: Option<Point>,
    provider: L
}

impl<L: NextLink> ChainIter<L> {
    pub fn new(starting_point: Point, provider: L) -> Self {
        let previous_point = None;

        Self { starting_point, previous_point, provider }
    }
}

impl<L: NextLink> Iterator for ChainIter<L> {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(previous_point) = self.previous_point {
            let next_point = self.provider.next_link(previous_point);

            if next_point != self.starting_point {
                self.previous_point = Some(next_point);

                Some(next_point)
            } else {
                None
            }
        } else {
            self.previous_point = Some(self.starting_point);

            Some(self.starting_point)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeOneNextLink;

    impl NextLink for FakeOneNextLink {
        fn next_link(&self, point: Point) -> Point {
            point
        }
    }

    #[test]
    fn single() {
        let point = Point::new(1, 1);

        assert_eq!(
            ChainIter::new(point, FakeOneNextLink {}).collect::<Vec<_>>(),
            vec! [point]
        );
    }

    struct FakeCycleNextLink {
        cycle: Vec<Point>
    }

    impl NextLink for FakeCycleNextLink {
        fn next_link(&self, point: Point) -> Point {
            if let Some(i) = self.cycle.iter().position(|&p| p == point) {
                let j = (i + 1) % self.cycle.len();

                self.cycle[j]
            } else {
                point
            }
        }
    }

    #[test]
    fn chain() {
        let cycle = vec! [
            Point::new(0, 0),
            Point::new(1, 1),
            Point::new(2, 2),
            Point::new(3, 3),
            Point::new(4, 4),
            Point::new(5, 5),
        ];

        assert_eq!(
            ChainIter::new(Point::new(2, 2), FakeCycleNextLink { cycle }).collect::<Vec<_>>(),
            vec! [Point::new(2, 2), Point::new(3, 3), Point::new(4, 4), Point::new(5, 5), Point::new(0, 0), Point::new(1, 1)]
        );
    }
}
