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
use dg_utils::{max, min};

use super::benson::BensonImpl;
use super::ladder::Ladder;
use super::symmetry;

/// The number of features that the board will provide.
pub const NUM_FEATURES: usize = 40;

/// The total size (in elements) of the feature set.
pub const FEATURE_SIZE: usize = NUM_FEATURES * 361;

/// Utility function for determining the data format of the array returned by
/// `get_features`.
pub trait Order {
    fn index(fs: usize, c: usize, point: Point) -> usize;
}

/// Implementation of `Order` for the data format `NCHW`.
pub struct CHW;

impl Order for CHW {
    fn index(_fs: usize, c: usize, point: Point) -> usize {
        c * 361 + point.to_packed_index()
    }
}

/// Implementation of `Order` for the data format `NHWC`.
pub struct HWC;

impl Order for HWC {
    fn index(nf: usize, c: usize, point: Point) -> usize {
        nf * point.to_packed_index() + c
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
    /// 12. Our liberties (>= 7)
    /// 13. Our liberties (>= 8)
    /// 14. Our liberties after move (>= 1)
    /// 15. Our liberties after move (>= 2)
    /// 16. Our liberties after move (>= 3)
    /// 17. Our liberties after move (>= 4)
    /// 18. Our liberties after move (>= 5)
    /// 19. Our liberties after move (>= 6)
    /// 20. Our liberties after move (>= 7)
    /// 21. Our liberties after move (>= 8)
    /// 22. Opponent liberties (>= 1)
    /// 23. Opponent liberties (>= 2)
    /// 24. Opponent liberties (>= 3)
    /// 25. Opponent liberties (>= 4)
    /// 26. Opponent liberties (>= 5)
    /// 27. Opponent liberties (>= 6)
    /// 28. Opponent liberties (>= 7)
    /// 29. Opponent liberties (>= 8)
    /// 30. Opponent liberties after move (>= 1)
    /// 31. Opponent liberties after move (>= 2)
    /// 32. Opponent liberties after move (>= 3)
    /// 33. Opponent liberties after move (>= 4)
    /// 34. Opponent liberties after move (>= 5)
    /// 35. Opponent liberties after move (>= 6)
    /// 36. Opponent liberties after move (>= 7)
    /// 37. Opponent liberties after move (>= 8)
    ///
    /// ## Vertex properties
    ///
    /// 38. Is super-ko
    /// 39. Is ladder capture
    /// 40. Is ladder escape
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

        let mut features = vec! [c_0; FEATURE_SIZE];
        let symmetry_table = symmetry.get_table();
        let opponent = to_move.opposite();

        // board state (one-hot historic)
        for (i, point) in self.board.history.iter().take(2).enumerate() {
            if point != Point::default() {
                let other = symmetry_table[point];

                features[O::index(NUM_FEATURES, 3+i, other)] = c_1;
            }
        }

        // liberties
        for index in Point::all() {
            let other = symmetry_table[index];

            if self.board.inner[index].color() != None {
                let start = if self.board.inner[index].color() == Some(to_move) { 5 } else { 21 };
                let num_liberties = ::std::cmp::min(
                    self.board.inner.get_n_liberty(index),
                    8
                );

                for i in 0..num_liberties {
                    features[O::index(NUM_FEATURES, start+i, other)] = c_1;
                }
            } else {
                if self.board.inner.is_valid(to_move, index) {
                    let num_liberties = ::std::cmp::min(
                        self.board.inner.get_n_liberty_if(to_move, index),
                        8
                    );

                    for i in 0..num_liberties {
                        features[O::index(NUM_FEATURES, 13+i, other)] = c_1;
                    }
                }

                if self.board.inner.is_valid(opponent, index) {
                    let num_liberties = ::std::cmp::min(
                        self.board.inner.get_n_liberty_if(opponent, index),
                        8
                    );

                    for i in 0..num_liberties {
                        features[O::index(NUM_FEATURES, 29+i, other)] = c_1;
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

                    features[O::index(NUM_FEATURES, 37, other)] = c_1;
                }

                // is ladder capture
                if self.board.inner.is_ladder_capture(to_move, index) {
                    features[O::index(NUM_FEATURES, 38, other)] = c_1;
                }

                // is ladder escape
                if self.board.inner.is_ladder_escape(to_move, index) {
                    features[O::index(NUM_FEATURES, 39, other)] = c_1;
                }
            }
        }

        // global properties
        let c_komi = T::from(max(min(0.5 + (0.5 * self.board.komi) / 7.5, 1.0), 0.0));

        let is_black = if to_move == Color::Black { c_komi } else { c_0 };
        let is_white = if to_move == Color::White { c_komi } else { c_0 };

        for index in Point::all() {
            let other = symmetry_table[index];

            features[O::index(NUM_FEATURES, 0, other)] = is_black;
            features[O::index(NUM_FEATURES, 1, other)] = is_white;
            features[O::index(NUM_FEATURES, 2, other)] = is_ko;
        }

        features
    }
}

pub struct LzFeatures<'a> {
    boards: Vec<&'a Board>
}

impl<'a> LzFeatures<'a> {
    pub fn new(boards: Vec<&'a Board>) -> Self {
        assert!(boards.len() <= 8);

        Self { boards }
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

        let mut out = vec! [c_0; 6498];
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
                    out[O::index(18, 0+i, other)] = if board.inner[point].color() == Some(to_move) { c_1 } else { c_0 };
                    out[O::index(18, 8+i, other)] = if board.inner[point].color() == Some(opposite) { c_1 } else { c_0 };
                }
            }

            /*
            17) All 1 if black is to move, 0 otherwise
            18) All 1 if white is to move, 0 otherwise
            */
            out[O::index(18, 16, other)] = is_black;
            out[O::index(18, 17, other)] = is_white;
        }

        out
    }
}

pub struct V2<'a> {
    board: &'a Board
}

impl<'a> V2<'a> {
    fn new(board: &'a Board) -> Self {
        Self { board }
    }

    fn self_komi(&self, to_move: Color) -> f32 {
        let komi =
            match to_move {
                Color::Black => -self.board.komi(),
                Color::White => self.board.komi()
            };

        (komi / 7.5).max(1.0).min(-1.0)
    }
}

impl<'a> Features for V2<'a> {
    fn get_features<O: Order, T: From<f32> + Copy>(
        &self,
        to_move: Color,
        symmetry: symmetry::Transform
    ) -> Vec<T>
    {
        let c_0 = T::from(0.0);
        let c_1 = T::from(1.0);
        let c_komi = T::from(self.self_komi(to_move));

        let mut out = vec! [c_0; 16 * 361];
        let symmetry_table = symmetry.get_table();
        let opponent = to_move.opposite();
        let benson_our = BensonImpl::new(self.board, to_move);
        let benson_opp = BensonImpl::new(self.board, opponent);

        for point in Point::all() {
            let other = symmetry_table[point];

            out[O::index(16, 0, other)] = c_1;
            out[O::index(16, 1, other)] = c_komi;

            if self.board.at(point) == Some(to_move) {
                out[O::index(16, 2, other)] = c_1;
            } else if self.board.at(point) == Some(opponent) {
                out[O::index(16, 3, other)] = c_1;
            } else if self.board.inner.is_valid(to_move, point) {
                out[O::index(16, 10, other)] = c_1;

                if self.board._is_ko(to_move, point) {
                    out[O::index(16, 13, other)] = c_1;
                }
            }

            if self.board.at(point) != None {
                match self.board.inner.get_n_liberty(point) {
                    1 => out[O::index(16, 4, other)] = c_1,
                    2 => out[O::index(16, 5, other)] = c_1,
                    3 => out[O::index(16, 6, other)] = c_1,
                    4 => out[O::index(16, 7, other)] = c_1,
                    _ => ()
                }
            } else if benson_our.is_eye(point) {
                out[O::index(16, 11, other)] = c_1;
            } else if benson_opp.is_eye(point) {
                out[O::index(16, 12, other)] = c_1;
            }

            if self.board.at(point) == None {
                match self.board.inner.get_n_liberty_if(to_move, point) {
                    1 => out[O::index(16, 8, other)] = c_1,
                    2 => out[O::index(16, 9, other)] = c_1,
                    _ => ()
                }
            }

            if self.board.inner.is_ladder_capture(to_move, point) {
                out[O::index(16, 14, other)] = c_1;
            }

            if self.board.inner.is_ladder_escape(to_move, point) {
                out[O::index(16, 15, other)] = c_1;
            }
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

        assert_eq!(features.len(), FEATURE_SIZE);
    }

    #[test]
    fn check_features_hwc() {
        let board = Board::new(0.5);
        let features = V1::new(&board)
            .get_features::<HWC, f32>(Color::Black, symmetry::Transform::Identity);

        assert_eq!(features.len(), FEATURE_SIZE);
    }
}
