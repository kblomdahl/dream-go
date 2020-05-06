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

use board_fast::BoardFast;
use color::Color;
use iter::{AdjacentChainIter, ValidIter, IsPartOf};
use point_state::Vertex;
use point::Point;

pub struct HasColor<'a> {
    board: &'a BoardFast,
    color: Option<Color>
}

impl<'a> HasColor<'a> {
    pub fn new(board: &'a BoardFast, color: Option<Color>) -> Self {
        Self { board, color }
    }
}

impl<'a> IsPartOf for HasColor<'a> {
    fn is_part_of(&self, point: Point) -> bool {
        let v = self.board[point];

        v.is_valid() && v.color() == self.color
    }
}

pub type LibertyIter<'a, L> = ValidIter<AdjacentChainIter<L>, HasColor<'a>>;

#[cfg(test)]
mod tests {
    use board::Board;
    use super::*;

    #[test]
    fn corner() {
        let mut board = Board::new(0.5);
        board.place(Color::Black, Point::new(0, 0));

        assert_eq!(
            board.inner.liberties_of(Point::new(0, 0)).collect::<Vec<_>>(),
            vec! [Point::new(1, 0), Point::new(0, 1)]
        );
    }

    #[test]
    fn middle() {
        let mut board = Board::new(0.5);
        board.place(Color::Black, Point::new(0, 1));
        board.place(Color::Black, Point::new(1, 0));
        board.place(Color::Black, Point::new(1, 1));

        assert_eq!(
            board.inner.liberties_of(Point::new(1, 1)).collect::<Vec<_>>(),
            vec! [
                Point::new(2, 1),
                Point::new(1, 2),
                Point::new(0, 0),
                Point::new(0, 2),
                Point::new(2, 0),
                Point::new(0, 0),
            ]
        );
    }
}
