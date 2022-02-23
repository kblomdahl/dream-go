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
    /// Returns the features of the current player in the given `order`, data
    /// type, and `symmetry`.
    ///
    /// # Arguments
    ///
    /// * `symmetry` - the symmetry to use
    ///
    fn get_features<O: Order, T: From<f32> + Copy>(
        &self,
        symmetry: symmetry::Transform
    ) -> Vec<T>;

    /// Returns the motion features for the most recent move in the given
    /// `order`, data type, and `symmetry`.
    ///
    /// # Arguments
    ///
    /// * `symmetry` - the symmetry to use
    ///
    fn get_motion_features<O: Order, T: From<f32> + Copy>(
        &self,
        symmetry: symmetry::Transform
    ) -> Vec<T>;

    /// Returns additional features that the network should be able to predict.
    ///
    /// # Arguments
    ///
    /// * `symmetry` - the symmetry to use
    ///
    fn get_additional_features<O: Order, T: From<f32> + Copy>(
        &self,
        symmetry: symmetry::Transform
    ) -> Vec<T>;
}

pub struct V1<'a> {
    board: &'a Board,
    to_move: Color
}

impl<'a> V1<'a> {
    pub fn new(to_move: Color, board: &'a Board) -> Self {
        Self { to_move, board }
    }

    /// Returns the number of channels.
    pub const fn num_features() -> usize {
        16
    }

    /// Returns the number of channels in the motion features.
    pub const fn num_motion_features() -> usize {
        8
    }

    /// Returns the number of channels in the motion features.
    pub const fn num_additional_features() -> usize {
        8
    }

    /// Returns the total number of elements that the returned features will
    /// contain.
    pub const fn size() -> usize {
        Self::num_features() * 361
    }
}

impl<'a> Features for V1<'a> {
    /// Returns the following minimal features for the current board state:
    ///
    /// ## Global properties
    ///
    ///  1. A constant plane filled with ones if we are black
    ///  2. A constant plane filled with ones if we are white
    ///  3. Komi (between -1 and +1)
    ///
    /// ## Board state
    ///
    ///  4. side to move stones
    ///  5. opponent stones
    ///
    /// ## History
    ///
    ///  6. stone played by side to play at T=0
    ///  7. stone played by opponent at T=0
    ///  8. stone played by side to play at T=-1
    ///  9. stone played by opponent at T=-1
    /// 10. stone played by side to play at T=-2
    /// 11. stone played by opponent at T=-2
    /// 12. stone played by side to play at T=-3
    /// 13. stone played by opponent at T=-3
    /// 14. stone played by side to play at T=-4
    /// 15. stone played by opponent at T=-4
    ///
    /// # Arguments
    ///
    /// * `to_move` - the color of the current player
    /// * `symmetry` - the symmetry to extract the features to
    ///
    fn get_features<O: Order, T: From<f32> + Copy>(
        &self,
        symmetry: symmetry::Transform
    ) -> Vec<T>
    {
        let c_0 = T::from(0.0);
        let c_1 = T::from(1.0);
        let o = O::new(Self::num_features());

        let mut out = vec! [c_0; Self::size()];
        let symmetry_table = symmetry.get_table();

        // board state
        for point in Point::all() {
            let other = symmetry_table[point];

            match self.board.at(point) {
                Some(color) if color == self.to_move => { out[o.index(3, other)] = c_1 },
                Some(color) if color != self.to_move => { out[o.index(4, other)] = c_1 },
                _ => {}
            }
        }

        // one-hot historic
        for (i, (played, point)) in self.board.history.iter().take(5).enumerate() {
            if point != Point::default() {
                let other = symmetry_table[point];

                out[o.index(2*i+5, other)] = if played == self.to_move { c_1 } else { c_0 };
                out[o.index(2*i+6, other)] = if played != self.to_move { c_1 } else { c_0 };
            }
        }

        // global properties
        let c_komi = T::from((0.5 + (0.5 * self.board.komi) / 7.5).min(1.0).max(0.0));

        let is_black = if self.to_move == Color::Black { c_1 } else { c_0 };
        let is_white = if self.to_move == Color::White { c_1 } else { c_0 };

        for index in Point::all() {
            let other = symmetry_table[index];

            out[o.index(0, other)] = is_black;
            out[o.index(1, other)] = is_white;
            out[o.index(2, other)] = c_komi;
        }

        out
    }

    /// Returns the following minimal motion / action features for the current
    /// board state:
    ///
    /// ## Global properties
    ///
    ///  1. A constant plane filled with ones if we are black
    ///  2. A constant plane filled with ones if we are white
    ///
    /// ## History
    ///
    ///  3. stone played by side to play at T=0
    ///  4. stone played by opponent at T=0
    ///  5. stone played by side to play at T=-1
    ///  6. stone played by opponent at T=-1
    ///  7. stone played by side to play at T=-2
    ///  8. stone played by opponent at T=-2
    ///
    fn get_motion_features<O: Order, T: From<f32> + Copy>(
        &self,
        symmetry: symmetry::Transform
    ) -> Vec<T>
    {
        let c_0 = T::from(0.0);
        let c_1 = T::from(1.0);
        let o = O::new(Self::num_features());

        let mut out = vec! [c_0; Self::size()];
        let symmetry_table = symmetry.get_table();

        for (i, (played, point)) in self.board.history.iter().take(3).enumerate() {
            if point != Point::default() {
                let other = symmetry_table[point];

                out[o.index(2*i+2, other)] = if played == self.to_move { c_1 } else { c_0 };
                out[o.index(2*i+3, other)] = if played != self.to_move { c_1 } else { c_0 };
            }
        }

        let is_black = if self.to_move == Color::Black { c_1 } else { c_0 };
        let is_white = if self.to_move == Color::White { c_1 } else { c_0 };

        for point in Point::all() {
            let other = symmetry_table[point];

            out[o.index(0, other)] = is_black;
            out[o.index(1, other)] = is_white;
        }

        out
    }

    /// Returns additional features for the current board state that the network
    /// should be able to predict from the given board state.
    ///
    /// ## Board state
    ///
    ///  1. side to move stones
    ///  2. opponent stones
    ///
    /// ## Vertex properties
    ///
    ///  3. is valid move
    ///  4. is ladder capture
    ///  5. is ladder escape
    ///  6. is eye for side to move
    ///  7. side to move stones is unconditionally alive
    ///  8. opponent stones is unconditionally alive
    ///
    fn get_additional_features<O: Order, T: From<f32> + Copy>(
        &self,
        symmetry: symmetry::Transform
    ) -> Vec<T>
    {
        let c_0 = T::from(0.0);
        let c_1 = T::from(1.0);
        let o = O::new(Self::num_additional_features());
        let symmetry_table = symmetry.get_table();
        let black_benson = BensonImpl::new(self.board, self.to_move);
        let white_benson = BensonImpl::new(self.board, self.to_move.opposite());
        let mut out = vec! [c_0; 361 * Self::num_additional_features()];

        for point in Point::all() {
            let other = symmetry_table[point];

            match self.board.at(point) {
                Some(color) if color == self.to_move => { out[o.index(0, other)] = c_1 },
                Some(color) if color != self.to_move => { out[o.index(1, other)] = c_1 },
                _ => {}
            }

            if self.board.is_valid(self.to_move, point) && !white_benson.is_eye(point) {
                out[o.index(2, other)] = c_1;
                out[o.index(3, other)] = if self.board.inner.is_ladder_capture(self.to_move, point) { c_1 } else { c_0 };
                out[o.index(4, other)] = if self.board.inner.is_ladder_escape(self.to_move, point) { c_1 } else { c_0 };
                out[o.index(5, other)] = if black_benson.is_eye(point) { c_1 } else { c_0 };
                out[o.index(6, other)] = if black_benson.is_alive(point) { c_1 } else { c_0 };
                out[o.index(7, other)] = if white_benson.is_alive(point) { c_1 } else { c_0 };
            }
        }

        out
    }
}

/// The features that the leela-zero project uses.
pub struct LzFeatures<'a> {
    boards: Vec<&'a Board>,
    to_move: Color
}

impl<'a> LzFeatures<'a> {
    pub fn new(to_move: Color, boards: Vec<&'a Board>) -> Self {
        assert!(boards.len() <= 8);

        Self { to_move, boards }
    }

    /// Returns the number of channels.
    pub const fn num_features() -> usize {
        18
    }

    /// Returns the number of channels in the motion features.
    pub const fn num_motion_features() -> usize {
        18
    }

    /// Returns the number of additional features channels
    pub const fn num_additional_features() -> usize {
        2
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
        symmetry: symmetry::Transform
    ) -> Vec<T>
    {
        let c_0 = T::from(0.0);
        let c_1 = T::from(1.0);
        let o = O::new(Self::num_features());

        let mut out = vec! [c_0; Self::size()];
        let symmetry_table = symmetry.get_table();
        let to_move = self.to_move;
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

    fn get_motion_features<O: Order, T: From<f32> + Copy>(
        &self,
        symmetry: symmetry::Transform
    ) -> Vec<T>
    {
        self.get_features::<O, T>(symmetry)
    }


    /// Returns additional features for the current board state that the network
    /// should be able to predict from the given board state.
    ///
    /// ## Board state
    ///
    ///  1. side to move stones
    ///  2. opponent stones
    ///
    fn get_additional_features<O: Order, T: From<f32> + Copy>(
        &self,
        symmetry: symmetry::Transform
    ) -> Vec<T>
    {
        let c_0 = T::from(0.0);
        let c_1 = T::from(1.0);
        let o = O::new(Self::num_additional_features());
        let symmetry_table = symmetry.get_table();

        let mut out = vec! [c_0; 361 * Self::num_additional_features()];
        let board = self.boards.last().unwrap();

        for point in Point::all() {
            let other = symmetry_table[point];

            match board.at(point) {
                Some(color) if color == self.to_move => { out[o.index(0, other)] = c_1 },
                Some(color) if color != self.to_move => { out[o.index(1, other)] = c_1 },
                _ => {}
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
        let features = V1::new(Color::Black, &board)
            .get_features::<CHW, f32>(symmetry::Transform::Identity);

        assert_eq!(features.len(), V1::size());
    }

    #[test]
    fn check_features_hwc() {
        let board = Board::new(0.5);
        let features = V1::new(Color::Black, &board)
            .get_features::<HWC, f32>(symmetry::Transform::Identity);

        assert_eq!(features.len(), V1::size());
    }

    #[test]
    fn check_v1_motion_features() {
        let board = Board::new(0.5);
        let features = V1::new(Color::Black, &board)
            .get_motion_features::<HWC, f32>(symmetry::Transform::Identity);

        assert_eq!(features.len(), V1::size());
    }

    #[test]
    fn check_v1_additional_features() {
        let board = Board::new(0.5);
        let features = V1::new(Color::Black, &board)
            .get_additional_features::<HWC, f32>(symmetry::Transform::Identity);

        assert_eq!(features.len(), 361 * V1::num_additional_features());
    }
}
