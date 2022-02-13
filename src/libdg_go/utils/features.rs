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

use board::Board;
use color::Color;
use point::Point;
use point_state::Vertex;

use super::benson::BensonImpl;
use super::ladder::Ladder;
use super::symmetry;

/// The default features to use internally.
pub type Default<'a> = self::V1<'a>;

/// Utility function for determining the data format of the array returned by
/// `get_features`.
pub trait Order {
    fn new(num_features: usize) -> Self;
    fn index(&self, c: usize, point: Point) -> usize;
}

/// Implementation of `Order` for the data format `NCHW`.
pub struct CHW;

impl Order for CHW {
    fn new(_num_features: usize) -> Self {
        Self {}
    }

    fn index(&self, c: usize, point: Point) -> usize {
        c * 361 + point.to_packed_index()
    }
}

/// Implementation of `Order` for the data format `NHWC`.
pub struct HWC {
    num_features: usize
}

impl Order for HWC {
    fn new(num_features: usize) -> Self {
        Self { num_features }
    }

    fn index(&self, c: usize, point: Point) -> usize {
        self.num_features * point.to_packed_index() + c
    }
}

pub trait Features {
    /// Returns the features of the current object in the given order and data
    /// type.
    ///
    /// # Arguments
    ///
    /// * `to_move` - the color of the current player
    /// * `symmetry` - the symmetry to use
    ///
    fn get_features<O: Order, T: From<f32> + Copy>(
        &self,
        to_move: Color,
        symmetry: symmetry::Transform
    ) -> Vec<T>;
}

pub struct V1<'a> {
    board: &'a Board
}

impl<'a> V1<'a> {
    pub fn new(board: &'a Board) -> Self {
        Self { board }
    }

    /// Returns the number of channels.
    pub const fn num_features() -> usize {
        32
    }

    /// Returns the total number of elements that the returned features will
    /// contain.
    pub const fn size() -> usize {
        Self::num_features() * 361
    }
}

impl<'a> Features for V1<'a> {
    /// Returns the features of the current board state for the given color,
    /// it returns the following features. Divided into four sections based
    /// on their intended purpose (regardless of what the network does with
    /// them).
    ///
    /// ## Global properties
    ///
    ///  1. A constant plane filled with ones if we are black
    ///  2. A constant plane filled with ones if we are white
    ///  3. A constant plane filled with ones if any move is super-ko
    ///
    /// ## One-hot historic board state
    ///
    ///  4. Most recent move ( 0)
    ///  5. Most recent move (-1)
    ///
    /// ## Liberties
    ///
    ///  6. Our liberties (>= 1)
    ///  7. Our liberties (>= 2)
    ///  8. Our liberties (>= 3)
    ///  9. Our liberties (>= 4)
    /// 10. Our liberties (>= 5)
    /// 11. Our liberties (>= 6)
    /// 12. Our liberties after move (>= 1)
    /// 13. Our liberties after move (>= 2)
    /// 14. Our liberties after move (>= 3)
    /// 15. Our liberties after move (>= 4)
    /// 16. Our liberties after move (>= 5)
    /// 17. Our liberties after move (>= 6)
    /// 18. Opponent liberties (>= 1)
    /// 19. Opponent liberties (>= 2)
    /// 20. Opponent liberties (>= 3)
    /// 21. Opponent liberties (>= 4)
    /// 22. Opponent liberties (>= 5)
    /// 23. Opponent liberties (>= 6)
    /// 24. Opponent liberties after move (>= 1)
    /// 25. Opponent liberties after move (>= 2)
    /// 26. Opponent liberties after move (>= 3)
    /// 27. Opponent liberties after move (>= 4)
    /// 28. Opponent liberties after move (>= 5)
    /// 29. Opponent liberties after move (>= 6)
    ///
    /// ## Vertex properties
    ///
    /// 30. Is super-ko
    /// 31. Is ladder capture
    /// 32. Is ladder escape
    ///
    /// # Arguments
    ///
    /// * `to_move` - the color of the current player
    /// * `symmetry` - the symmetry to extract the features to
    ///
    fn get_features<O: Order, T: From<f32> + Copy>(
        &self,
        to_move: Color,
        symmetry: symmetry::Transform
    ) -> Vec<T>
    {
        let c_0 = T::from(0.0);
        let c_1 = T::from(1.0);
        let o = O::new(Self::num_features());

        let mut features = vec! [c_0; Self::size()];
        let symmetry_table = symmetry.get_table();
        let opponent = to_move.opposite();

        // board state (one-hot historic)
        for (i, point) in self.board.history.iter().take(2).enumerate() {
            if point != Point::default() {
                let other = symmetry_table[point];

                features[o.index(3+i, other)] = c_1;
            }
        }

        // liberties
        for index in Point::all() {
            let other = symmetry_table[index];

            if self.board.inner[index].color() != None {
                let start = if self.board.inner[index].color() == Some(to_move) { 5 } else { 17 };
                let num_liberties = self.board.inner.get_n_liberty(index).min(6);

                for i in 0..num_liberties {
                    features[o.index(start+i, other)] = c_1;
                }
            } else {
                if self.board.inner.is_valid(to_move, index) {
                    let num_liberties = self.board.inner.get_n_liberty_if(to_move, index).min(6);

                    for i in 0..num_liberties {
                        features[o.index(11+i, other)] = c_1;
                    }
                }

                if self.board.inner.is_valid(opponent, index) {
                    let num_liberties = self.board.inner.get_n_liberty_if(opponent, index).min(6);

                    for i in 0..num_liberties {
                        features[o.index(23+i, other)] = c_1;
                    }
                }
            }
        }

        // vertex properties
        let mut is_ko = c_0;

        for index in Point::all() {
            let other = symmetry_table[index];

            if self.board.inner[index].color() != None {
                // pass
            } else if self.board.inner.is_valid(to_move, index) {
                // is super-ko
                if self.board._is_ko(to_move, index) {
                    is_ko = c_1;

                    features[o.index(29, other)] = c_1;
                }

                // is ladder capture
                if self.board.inner.is_ladder_capture(to_move, index) {
                    features[o.index(30, other)] = c_1;
                }

                // is ladder escape
                if self.board.inner.is_ladder_escape(to_move, index) {
                    features[o.index(31, other)] = c_1;
                }
            }
        }

        // global properties
        let c_komi = T::from((0.5 + (0.5 * self.board.komi) / 7.5).min(1.0).max(0.0));

        let is_black = if to_move == Color::Black { c_komi } else { c_0 };
        let is_white = if to_move == Color::White { c_komi } else { c_0 };

        for index in Point::all() {
            let other = symmetry_table[index];

            features[o.index(0, other)] = is_black;
            features[o.index(1, other)] = is_white;
            features[o.index(2, other)] = is_ko;
        }

        features
    }
}

pub struct V2<'a> {
    board: &'a Board
}

impl<'a> V2<'a> {
    pub fn new(board: &'a Board) -> Self {
        Self { board }
    }

    /// Returns the number of channels.
    pub const fn num_features() -> usize {
        32
    }

    /// Returns the total number of elements that the returned features will
    /// contain.
    pub const fn size() -> usize {
        Self::num_features() * 361
    }

    fn self_komi(&self, to_move: Color) -> f32 {
        let komi =
            match to_move {
                Color::Black => -self.board.komi(),
                Color::White => self.board.komi()
            };

        (komi / 7.5).clamp(-1.0, 1.0)
    }

    #[inline]
    fn fill_color_specific<O: Order, T: From<f32> + Copy>(
        &self,
        offset: usize,
        to_move: Color,
        symmetry_table: &[Point],
        out: &mut [T]
    )
    {
        let benson = BensonImpl::new(self.board, to_move);
        let c_1 = T::from(1.0);
        let o = O::new(Self::num_features());

        for point in Point::all() {
            let other = symmetry_table[point];

            // Player
            let is_valid = benson.is_valid(point) && self.board.inner.is_valid(to_move, point);

            if is_valid {
                out[o.index(offset+2, other)] = c_1; // is valid move

                if self.board.inner.is_ladder_capture(to_move, point) {
                    out[o.index(offset+0, other)] = c_1; // is ladder capture
                }
                if self.board.inner.is_ladder_escape(to_move, point) {
                    out[o.index(offset+1, other)] = c_1; // is ladder escape
                }
                if benson.is_eye(point) {
                    out[o.index(offset+3, other)] = c_1; // is eye
                }
            }

            // Player Liberties
            if self.board.at(point) == Some(to_move) {
                let n = self.board.inner.get_n_liberty(point);

                for i in 0..n.max(4) {
                    out[o.index(offset+4+i, other)] = c_1;
                }
            } else if is_valid {
                let n = self.board.inner.get_n_liberty_if(to_move, point);

                for i in 0..n.max(4) {
                    out[o.index(offset+8+i, other)] = c_1;
                }
            }
        }
    }
}

impl<'a> Features for V2<'a> {
    /// Returns the features of the current board state for the given color,
    /// it returns the following features.
    ///
    /// ## Global properties
    ///
    ///  1. A constant plane filled with ones if we are black
    ///  2. A constant plane filled with ones if we are white
    ///  3. Komi (-1 to +1)
    ///
    /// ## One-hot historic board state
    ///
    ///  4. Most recent move ( 0)
    ///  5. Most recent move (-1)
    ///
    /// ## Vertex properties
    ///
    ///  6. Is corner
    ///  7. Is edge
    ///  8. Is middle
    ///
    /// ## Player
    ///
    ///  9. is ladder capture
    /// 10. is ladder escape
    /// 11. is valid move
    /// 12. is eye
    ///
    /// ### Player Liberties
    ///
    /// 13. liberties (>= 1)
    /// 14. liberties (>= 2)
    /// 15. liberties (>= 3)
    /// 16. liberties (>= 4)
    /// 17. liberties if played (>= 1)
    /// 18. liberties if played (>= 2)
    /// 19. liberties if played (>= 3)
    /// 20. liberties if played (>= 4)
    ///
    /// ## Player
    ///
    /// 21. is ladder capture
    /// 22. is ladder escape
    /// 23. is valid move
    /// 24. is eye
    ///
    /// ### Player Liberties
    ///
    /// 25. liberties (>= 1)
    /// 26. liberties (>= 2)
    /// 27. liberties (>= 3)
    /// 28. liberties (>= 4)
    /// 29. liberties if played (>= 1)
    /// 30. liberties if played (>= 2)
    /// 31. liberties if played (>= 3)
    /// 32. liberties if played (>= 4)
    ///
    /// # Arguments
    ///
    /// * `to_move` -
    /// * `symmetry`-
    ///
    fn get_features<O: Order, T: From<f32> + Copy>(
        &self,
        to_move: Color,
        symmetry: symmetry::Transform
    ) -> Vec<T>
    {
        let c_0 = T::from(0.0);
        let c_1 = T::from(1.0);
        let c_komi = T::from(self.self_komi(to_move));
        let o = O::new(Self::num_features());

        let mut out = vec! [c_0; Self::size()];
        let symmetry_table = symmetry.get_table();

        for point in Point::all() {
            let other = symmetry_table[point];

            // Global properties
            out[o.index(0, other)] = if to_move == Color::Black { c_1 } else { c_0 };
            out[o.index(1, other)] = if to_move == Color::White { c_1 } else { c_0 };
            out[o.index(2, other)] = c_komi;

            // Vertex properties
            if (point.x() == 0 || point.x() == 18) && (point.y() == 0 || point.y() == 18) {
                out[o.index(5, other)] = c_1; // corner
            } else if point.x() == 0 || point.x() == 18 || point.y() == 0 || point.y() == 18 {
                out[o.index(6, other)] = c_1; // edge
            } else {
                out[o.index(7, other)] = c_1; // middle
            }
        }

        self.fill_color_specific::<O, _>( 8, to_move, symmetry_table, &mut out);
        self.fill_color_specific::<O, _>(20, to_move.opposite(), symmetry_table, &mut out);

        // One-hot historic board state
        for (i, point) in self.board.history.iter().take(2).enumerate() {
            if point != Point::default() {
                let other = symmetry_table[point];

                out[o.index(3+i, other)] = c_1;
            }
        }

        out
    }
}

/// The features that the leela-zero project uses.
pub struct LzFeatures<'a> {
    boards: Vec<&'a Board>
}

impl<'a> LzFeatures<'a> {
    pub fn new(boards: Vec<&'a Board>) -> Self {
        assert!(boards.len() <= 8);

        Self { boards }
    }

    /// Returns the number of channels.
    pub const fn num_features() -> usize {
        18
    }

    /// Returns the total number of elements that the returned features will
    /// contain.
    pub const fn size() -> usize {
        Self::num_features() * 361
    }
}

impl<'a> Features for LzFeatures<'a> {
    fn get_features<O: Order, T: From<f32> + Copy>(
        &self,
        to_move: Color,
        symmetry: symmetry::Transform
    ) -> Vec<T>
    {
        let c_0 = T::from(0.0);
        let c_1 = T::from(1.0);
        let o = O::new(Self::num_features());

        let mut out = vec! [c_0; Self::size()];
        let symmetry_table = symmetry.get_table();
        let opposite = to_move.opposite();

        let is_black = if to_move == Color::Black { c_1 } else { c_0 };
        let is_white = if to_move == Color::White { c_1 } else { c_0 };

        for point in Point::all() {
            let other = symmetry_table[point];

            /*
            1) Side to move stones at time T=0
            2) Side to move stones at time T=-1  (0 if T=0)
            ...
            8) Side to move stones at time T=-7  (0 if T<=6)
            9) Other side stones at time T=0
            10) Other side stones at time T=-1   (0 if T=0)
            ...
            16) Other side stones at time T=-7   (0 if T<=6)
            */
            for i in 0..8 {
                if let Some(board) = self.boards.get(i) {
                    out[o.index(0+i, other)] = if board.inner[point].color() == Some(to_move) { c_1 } else { c_0 };
                    out[o.index(8+i, other)] = if board.inner[point].color() == Some(opposite) { c_1 } else { c_0 };
                }
            }

            /*
            17) All 1 if black is to move, 0 otherwise
            18) All 1 if white is to move, 0 otherwise
            */
            out[o.index(16, other)] = is_black;
            out[o.index(17, other)] = is_white;
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_features_chw() {
        let board = Board::new(0.5);
        let features = V1::new(&board)
            .get_features::<CHW, f32>(Color::Black, symmetry::Transform::Identity);

        assert_eq!(features.len(), V1::size());
    }

    #[test]
    fn check_features_hwc() {
        let board = Board::new(0.5);
        let features = V1::new(&board)
            .get_features::<HWC, f32>(Color::Black, symmetry::Transform::Identity);

        assert_eq!(features.len(), V1::size());
    }

    #[test]
    fn check_v2() {
        let mut board = Board::new(6.5);
        board.place(Color::Black, Point::new(3, 3));
        board.place(Color::White, Point::new(15, 15));

        let features = V2::new(&board)
            .get_features::<HWC, f32>(Color::Black, symmetry::Transform::Identity);

        assert_eq!(features.len(), V2::size());
    }
}
