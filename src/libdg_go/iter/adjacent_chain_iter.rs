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
use iter::{AdjacentIter, ChainIter, NextLink};

pub struct AdjacentChainIter<L: NextLink> {
    iter: ChainIter<L>,
    adj_iter: Option<AdjacentIter>
}

impl<L: NextLink> AdjacentChainIter<L> {
    pub fn new(iter: ChainIter<L>) -> Self {
        let adj_iter = None;

        Self { iter, adj_iter }
    }
}

impl<L: NextLink> Iterator for AdjacentChainIter<L> {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ref mut adj_iter) = self.adj_iter {
            let out = adj_iter.next();

            match out {
                None => {
                    self.adj_iter = None;
                    self.next()
                },
                Some(x) => Some(x),
            }
        } else if let Some(point) = self.iter.next() {
            self.adj_iter = Some(AdjacentIter::new(point));
            self.next()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeNextLink;

    impl NextLink for FakeNextLink {
        fn next_link(&self, point: Point) -> Point {
            if point == Point::new(1, 1) {
                Point::new(2, 1)
            } else {
                Point::new(1, 1)
            }
        }
    }

    #[test]
    fn corner() {
        let adj_chain = AdjacentChainIter::new(ChainIter::new(Point::new(1, 1), FakeNextLink {}));

        assert_eq!(
            adj_chain.collect::<Vec<_>>(),
            vec! [
                Point::new(2, 1),
                Point::new(1, 0),
                Point::new(0, 1),
                Point::new(1, 2),
                Point::new(3, 1),
                Point::new(2, 0),
                Point::new(1, 1),
                Point::new(2, 2),
            ]
        );
    }
}
