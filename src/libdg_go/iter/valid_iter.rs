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

pub trait IsValid {
    fn is_valid(&self, point: Point) -> bool;
}

/// Iterates over all points that are considered valid according to the given
/// validator.
pub struct ValidIter<I: Iterator<Item=Point>, T: IsValid> {
    iter: I,
    validator: T
}

impl<I: Iterator<Item=Point>, T: IsValid> ValidIter<I, T> {
    pub fn new(iter: I, validator: T) -> Self {
        Self { iter, validator }
    }
}

impl<I: Iterator<Item=Point>, T: IsValid> Iterator for ValidIter<I, T> {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(point) = self.iter.next() {
            if self.validator.is_valid(point) {
                return Some(point);
            }
        }

        None
    }
}

#[cfg(test)]
mod test {
    use super::*;

    struct FakeIsValid;

    impl IsValid for FakeIsValid {
        fn is_valid(&self, point: Point) -> bool {
            point.x() == 0
        }
    }

    #[test]
    fn keep_zeros() {
        let original = vec! [
            Point::new(1, 0),
            Point::new(0, 1),
            Point::new(2, 0),
            Point::new(0, 2),
            Point::new(3, 0),
            Point::new(0, 3)
        ];
        let expected = vec! [
            Point::new(0, 1),
            Point::new(0, 2),
            Point::new(0, 3)
        ];

        assert_eq!(
            ValidIter::new(original.into_iter(), FakeIsValid{}).collect::<Vec<_>>(),
            expected
        )
    }
}
