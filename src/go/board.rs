// Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use std::fmt;
use std::hash::{Hash, Hasher};

use go::board_fast::BoardFast;
use go::color::Color;
use go::circular_buf::CircularBuf;
use go::small_set::SmallSet;

/// 
#[derive(Clone)]
pub struct Board {
    /// The interior board representation.
    pub(super) inner: BoardFast,

    /// Stack containing the six most recent `vertices`.
    pub(super) history: CircularBuf,

    /// The komi used for this game.
    pub(super) komi: f32,

    /// The total number of moves that has been played on this board.
    pub(super) count: u16,

    /// The color of the player who played the most recent move.
    pub(super) last_played: Option<Color>,

    /// The zobrist hash of the current board state.
    pub(super) zobrist_hash: u64,

    /// The zobrist hash of the most recent board positions.
    pub(super) zobrist_history: SmallSet
}

impl Board {
    pub fn new(komi: f32) -> Board {
        Board {
            inner: BoardFast::new(),
            history: CircularBuf::new(),
            komi: komi,
            count: 0,
            last_played: None,
            zobrist_hash: 0,
            zobrist_history: SmallSet::new()
        }
    }

    /// Returns the width and height of this board.
    #[inline]
    pub fn size(&self) -> usize {
        19
    }

    /// Returns the komi of this board.
    #[inline]
    pub fn komi(&self) -> f32 {
        self.komi
    }

    /// Sets the komi of this board.
    #[inline]
    pub fn set_komi(&mut self, komi: f32) {
        self.komi = komi;
    }

    /// Returns the number of moves that has been played on this board.
    #[inline]
    pub fn count(&self) -> usize {
        self.count as usize
    }

    /// Returns the zobrist hash of this board.
    #[inline]
    pub fn zobrist_hash(&self) -> u64 {
        self.zobrist_hash
    }

    /// Returns the color of the last player that played a move.
    #[inline]
    pub fn last_played(&self) -> Option<Color> {
        self.last_played
    }

    /// Returns the color (if the vertex is not empty) of the stone at
    /// the given coordinates.
    ///
    /// # Arguments
    ///
    /// * `x` - the column of the coordinates
    /// * `y` - the row of the coordinates
    ///
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> Option<Color> {
        let index = 19 * y + x;

        if self.inner.vertices[index] == Color::Black as u8 {
            Some(Color::Black)
        } else if self.inner.vertices[index] == Color::White as u8 {
            Some(Color::White)
        } else {
            None
        }
    }

    /// Returns true if playing at the given index violated the
    /// super-ko rule.
    /// 
    /// # Arguments
    /// 
    /// * `color` - the color of the move
    /// * `index` - the index of the move
    /// 
    pub(super) fn _is_ko(&self, color: Color, index: usize) -> bool {
        debug_assert!(self.inner.is_valid(color, index));

        let adjust = self.inner.place_if(color, index);
        let next_zobrist_hash = self.zobrist_hash ^ adjust;

        self.zobrist_history.contains(next_zobrist_hash)
    }

    /// Returns whether the given move is valid according to the
    /// Tromp-Taylor rules.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `index` - the index of the move
    ///
    pub(super) fn _is_valid(&self, color: Color, index: usize) -> bool {
        self.inner.is_valid(color, index) && !self._is_ko(color, index)
    }

    /// Returns whether the given move is valid according to the
    /// Tromp-Taylor rules.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `x` - the column of the move
    /// * `y` - the row of the move
    ///
    pub fn is_valid(&self, color: Color, x: usize, y: usize) -> bool {
        self._is_valid(color, 19 * y + x)
    }

    /// Place the given stone on the board without checking if it is legal, the
    /// board is then updated according to the Tromp-Taylor rules with the
    /// except that ones own color is not cleared.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `index` - The index of the move
    ///
    pub(super) fn _place(&mut self, color: Color, index: usize) {
        // place the stone on the board regardless of whether it is legal
        // or not.
        self.zobrist_hash ^= self.inner.place(color, index);
        self.last_played = Some(color);
        self.count += 1;

        // store the actually played move since it is necessary for the feature
        // vector.
        self.history.push(index);
        self.zobrist_history.push(self.zobrist_hash);
    }

    /// Place the given stone on the board without checking if it is legal, the
    /// board is then updated according to the Tromp-Taylor rules with the
    /// except that ones own color is not cleared.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `x` - The column of the move
    /// * `y` - The row of the move
    ///
    pub fn place(&mut self, color: Color, x: usize, y: usize) {
        self._place(color, 19 * y + x)
    }
}

impl fmt::Display for Board {
    /// Pretty-print the current board in a similar format as the KGS client.
    ///
    /// # Arguments
    ///
    /// * `f` - the formatter to write the game to
    ///
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const LETTERS: [char; 25] = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z'
        ];

        write!(f, "    ")?;
        for i in 0..19 { write!(f, " {}", LETTERS[i])?; }
        write!(f, "\n")?;
        write!(f, "   \u{256d}")?;
        for _ in 0..19 { write!(f, "\u{2500}\u{2500}")?; }
        write!(f, "\u{2500}\u{256e}\n")?;

        for y in 0..19 {
            let y = 18 - y;

            write!(f, "{:2} \u{2502}", 1 + y)?;

            for x in 0..19 {
                let index = 19 * y + x;

                if self.inner.vertices[index] == 0 {
                    write!(f, "  ")?;
                } else if self.inner.vertices[index] == Color::Black as u8 {
                    write!(f, " \u{25cf}")?;
                } else if self.inner.vertices[index] == Color::White as u8 {
                    write!(f, " \u{25cb}")?;
                }
            }

            write!(f, " \u{2502} {}\n", 1 + y)?;
        }

        write!(f, "   \u{2570}")?;
        for _ in 0..19 { write!(f, "\u{2500}\u{2500}")?; }
        write!(f, "\u{2500}\u{256f}\n")?;
        write!(f, "    ")?;
        for i in 0..19 { write!(f, " {}", LETTERS[i])?; }
        write!(f, "\n")?;
        write!(f, "    \u{25cf} Black    \u{25cb} White\n")?;

        Ok(())
    }
}

impl Hash for Board {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // include the entire zobrist hash history, since we use six planes of
        // historic data in the features, and transposing them does not necessary
        // result in the same neural network output (mostly due to super-ko).
        for z in self.zobrist_history.iter() {
            state.write_u64(z);
        }

        state.write_u32(self.komi.to_bits());
    }
}

impl PartialEq for Board {
    fn eq(&self, other: &Board) -> bool {
        let history = self.zobrist_history.iter()
            .zip(other.zobrist_history.iter())
            .all(|(a, b)| a == b);

        history && self.inner.vertices.iter()
            .zip(other.inner.vertices.iter())
            .all(|(a, b)| a == b)
    }
}

impl Eq for Board { }

#[cfg(test)]
mod tests {
    use go::*;

    /// Test that it is possible to capture a stone in the middle of the
    /// board.
    #[test]
    fn capture() {
        let mut board = Board::new(DEFAULT_KOMI);

        board.place(Color::Black,  9,  9);
        board.place(Color::White,  8,  9);
        board.place(Color::White, 10,  9);
        board.place(Color::White,  9,  8);
        board.place(Color::White,  9, 10);

        assert_eq!(board.at(9, 9), None);
    }

    /// Test that it is possible to capture a group of stones in the corner.
    #[test]
    fn capture_group() {
        let mut board = Board::new(DEFAULT_KOMI);

        board.place(Color::Black, 0, 1);
        board.place(Color::Black, 1, 0);
        board.place(Color::Black, 0, 0);
        board.place(Color::Black, 1, 1);

        board.place(Color::White, 2, 0);
        board.place(Color::White, 2, 1);
        board.place(Color::White, 0, 2);
        board.place(Color::White, 1, 2);

        assert_eq!(board.at(0, 0), None);
        assert_eq!(board.at(0, 1), None);
        assert_eq!(board.at(1, 0), None);
        assert_eq!(board.at(1, 1), None);
    }

    /// Test that it is not possible to play a suicide move in the corner
    /// with two adjacent neighbours of the opposite color.
    #[test]
    fn suicide_corner() {
        let mut board = Board::new(DEFAULT_KOMI);

        board.place(Color::White, 0, 0);
        board.place(Color::Black, 1, 0);
        board.place(Color::Black, 0, 1);

        assert_eq!(board.at(0, 0), None);
        assert!(!board.is_valid(Color::White, 0, 0));
        assert!(board.is_valid(Color::Black, 0, 0));
    }

    /// Test that it is not possible to play a suicide move in the middle
    /// of a ponnuki.
    #[test]
    fn suicide_middle() {
        let mut board = Board::new(DEFAULT_KOMI);

        board.place(Color::Black,  9,  9);
        board.place(Color::White,  8,  9);
        board.place(Color::White, 10,  9);
        board.place(Color::White,  9,  8);
        board.place(Color::White,  9, 10);

        assert_eq!(board.at(9, 9), None);
        assert!(!board.is_valid(Color::Black, 9, 9));
        assert!(board.is_valid(Color::White, 9, 9));
    }

    /// Test that we can accurately detect ko using the simplest possible
    /// corner ko.
    #[test]
    fn ko() {
        let mut board = Board::new(DEFAULT_KOMI);

        board.place(Color::Black, 0, 0);
        board.place(Color::Black, 0, 2);
        board.place(Color::Black, 1, 1);
        board.place(Color::White, 1, 0);
        board.place(Color::White, 0, 1);

        assert!(!board.is_valid(Color::Black, 0, 0));
    }
}
