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

use board_fast::{BoardFast, One, Two, Three};
use color::Color;

pub trait Ladder {
    fn is_ladder_capture(&self, color: Color, index: usize) -> bool;
    fn is_ladder_escape(&self, color: Color, index: usize) -> bool;
}

/// Return true if the given group can capture any of its opponents
/// neighbouring groups.
///
/// # Arguments
///
/// * `board` - the `vertices` of the board to check
/// * `color` - the color of the current player
/// * `index` - the index of the group to check
///
fn _can_escape_with_capture(board: &BoardFast, color: Color, index: usize) -> bool {
    let mut current = index;
    let opponent = color.opposite() as u8;

    loop {
        foreach_4d!(board, current, |other_index, value| {
            if value == opponent && !board.has_n_liberty::<Two>(other_index, 2) {
                return true;
            }
        });

        current = unsafe { *board.next_vertex.get_unchecked(current) as usize };
        if current == index {
            break;
        }
    }

    false
}

/// Returns true if playing a stone at the given index successfully
/// captures some stones in a serie of ataris.
///
/// # Arguments
///
/// * `board` - the `vertices` of the board to check
/// * `color` - the color of the current player
/// * `index` - the index of the vertex to check
///
fn _is_ladder_capture(mut board: BoardFast, color: Color, index: usize) -> bool {
    board.place(color, index);

    // if any of the neighbouring opponent groups were reduced to one
    // liberty (and it cannot counter capture a group) then extend into
    // that liberty. if no such group exists then this is not a ladder
    // capturing move.
    let opponent = color.opposite() as u8;
    let opponent_index = find_4d!(board, index, |other_index, value| {
        if value == opponent {
            let is_in_atari = !board.has_n_liberty::<Two>(other_index, 2);

            if is_in_atari && !_can_escape_with_capture(&board, color.opposite(), other_index) {
                Some(board.get_n_liberty::<One>(other_index, 1))
            } else {
                None
            }
        } else {
            None
        }
    });

    if opponent_index.is_none() {
        return false
    }

    let opponent_index = opponent_index.unwrap();

    board.place(color.opposite(), opponent_index);

    // check the number of liberties after extending the group that was put in atari
    //
    // * If one liberty, then this group can be captured.
    // * If two liberties, keep searching.
    // * If more than two liberties, then this group can not be captured.
    //
    if !board.has_n_liberty::<Two>(opponent_index, 2) {
        return true;
    } else if board.has_n_liberty::<Three>(opponent_index, 3) {
        return false;
    }

    // if playing `opponent_vertex` put any of my stones into atari
    // then this is not a ladder capturing move.
    let player = color as u8;

    foreach_4d!(board, opponent_index, |other_index, value| {
        if value == player && !board.has_n_liberty::<Two>(other_index, 2) {
            return false;
        }
    });

    // try capturing the new group by playing _ladder capturing moves_
    // in all of its liberties, if we succeed with either then this
    // is a ladder capturing move
    foreach_4d!(board, opponent_index, |other_index, value| {
        if value == 0 {
            let other = board.clone();

            if _is_ladder_capture(other, color, other_index) {
                return true;
            }
        }
    });

    false
}

impl Ladder for BoardFast {
    /// Returns true if playing a stone at the given index allows us to
    /// capture some of the opponents stones with a ladder (sequence of
    /// ataris).
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the current player
    /// * `index` - the index of the stone to check
    ///
    fn is_ladder_capture(&self, color: Color, index: usize) -> bool {
        debug_assert!(self.is_valid(color, index));

        _is_ladder_capture(self.clone(), color, index)
    }

    /// Returns true if playing a stone at the given index allows us to
    /// escape using a ladder (sequence of ataris).
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the current player
    /// * `index` - the index of the stone to check
    fn is_ladder_escape(&self, color: Color, index: usize) -> bool {
        debug_assert!(self.is_valid(color, index));

        // check if we are connected to a stone with one liberty
        let player = color as u8;
        let connected_to_one = find_4d!(self, index, |other_index, value| {
            if value == player && !self.has_n_liberty::<Two>(other_index, 2) {
                Some(())
            } else {
                None
            }
        });

        if connected_to_one.is_none() {
            return false;
        }

        // clone only the minimum parts of the board that is necessary
        // to play out the ladder.
        let mut board = self.clone();

        board.place(color, index);

        // check if we have exactly two liberties
        let num_liberties = board.get_n_liberty::<Three>(index, 3).into_iter()
            .fold(0, |acc, &liberty| {
                acc + if liberty == 0xffff { 0 } else { 1 }
            });

        if num_liberties != 2 {
            return false;
        }

        // check that we cannot be captured in a ladder from either direction
        foreach_4d!(self, index, |other_index, value| {
            if value == 0 {
                let board = board.clone();

                if _is_ladder_capture(board, color.opposite(), other_index) {
                    return false;
                }
            }
        });

        true
    }
}

#[cfg(test)]
mod tests {
    use board::*;
    use color::*;
    use ladder::*;

    #[test]
    fn ladder_corner_capture() {
        // test the following (as 19x19 board), and check
        // that any atari move is a ladder capture
        //
        // X . . . X
        // . . . . .
        // . . . . .
        // . . . . .
        // X . . . X
        //
        let mut board = Board::new(7.5);
        board.place(Color::Black,  0,  0);
        board.place(Color::Black,  0, 18);
        board.place(Color::Black, 18,  0);
        board.place(Color::Black, 18, 18);

        for x in 0..19 {
            for y in 0..19 {
                if board.is_valid(Color::White, x, y) {
                    let is_ladder = (x == 1 && y == 0)
                        || (x ==  0 && y ==  1)
                        || (x == 18 && y == 17)
                        || (x == 17 && y == 18)
                        || (x ==  1 && y == 18)
                        || (x == 18 && y ==  1)
                        || (x ==  0 && y == 17)
                        || (x == 17 && y ==  0);
                    let index = 19 * y + x;

                    assert_eq!(
                        board.inner.is_ladder_capture(Color::White, index),
                        is_ladder
                    );
                }
            }
        }
    }

    #[test]
    fn ladder_capture() {
        // test that the most standard ladder capture is detected correctly:
        //
        // . . . . .
        // . . X X .
        // . X O . .
        // . . . . .
        // . . . . .
        //
        let mut board = Board::new(7.5);
        board.place(Color::White, 3, 3);
        board.place(Color::Black, 2, 3);
        board.place(Color::Black, 3, 2);
        board.place(Color::Black, 4, 2);

        for x in 0..19 {
            for y in 0..19 {
                if board.is_valid(Color::Black, x, y) {
                    let is_ladder = x == 3 && y == 4;
                    let index = 19 * y + x;

                    assert_eq!(
                        board.inner.is_ladder_capture(Color::Black, index),
                        is_ladder
                    );
                }
            }
        }
    }

    #[test]
    fn ladder_escape() {
        // test a standard ladder pattern with a stone on the diagonal
        let mut board = Board::new(7.5);
        board.place(Color::White,  3,  3);
        board.place(Color::White, 15, 15);  // ladder breaking
        board.place(Color::Black,  2,  3);
        board.place(Color::Black,  3,  2);
        board.place(Color::Black,  4,  2);
        board.place(Color::Black,  3,  4);

        for x in 0..19 {
            for y in 0..19 {
                if board.is_valid(Color::White, x, y) {
                    let index = 19 * y + x;

                    // check that nothing is a ladder capture
                    assert!(!board.inner.is_ladder_capture(Color::Black, index));

                    // check that only the one move is a ladder escape
                    let is_escape = x == 4 && y == 3;

                    assert_eq!(
                        board.inner.is_ladder_escape(Color::White, index),
                        is_escape,
                        "({}, {}) is a ladder escape = {}", x, y, is_escape
                    );
                }
            }
        }
    }

    /// Test a real games that once resulted in an infinite loop during ladder
    /// checks
    #[test]
    fn not_ladder() {
        let moves = [
            (Color::Black, 15,  3), (Color::White,  3, 15), (Color::Black, 16, 15), (Color::White,  3,  2),
            (Color::Black, 14, 16), (Color::White,  2,  4), (Color::Black,  3,  5), (Color::White,  3,  4),
            (Color::Black,  4,  5), (Color::White,  4,  4), (Color::Black,  5,  5), (Color::White,  1,  6),
            (Color::Black, 12,  2), (Color::White, 16,  9), (Color::Black, 16,  7), (Color::White, 16, 12),
            (Color::Black,  6,  3), (Color::White, 14,  9), (Color::Black, 15, 10), (Color::White, 15,  9),
            (Color::Black, 14,  7), (Color::White, 16,  2), (Color::Black, 16,  3), (Color::White, 15,  2),
            (Color::Black, 14,  2), (Color::White, 14,  1), (Color::Black, 13,  1), (Color::White, 14,  3),
            (Color::Black, 13,  2), (Color::White, 17,  3), (Color::Black, 17,  4), (Color::White, 17,  1),
            (Color::Black, 18,  3), (Color::White, 17,  2), (Color::Black, 14,  4), (Color::White, 15,  1),
            (Color::Black, 17, 11), (Color::White, 17, 12), (Color::Black, 17, 10), (Color::White, 15, 11),
            (Color::Black, 17,  9), (Color::White, 14, 10), (Color::Black, 12,  8), (Color::White,  5, 16),
            (Color::Black, 12, 10), (Color::White, 14, 14), (Color::Black, 12, 15), (Color::White, 12, 14),
            (Color::Black, 11, 14), (Color::White, 12, 13), (Color::Black, 11, 15), (Color::White, 15, 16),
            (Color::Black, 15, 15), (Color::White, 13, 16), (Color::Black, 13, 15), (Color::White, 14, 15),
            (Color::Black, 14, 17), (Color::White, 15, 17), (Color::Black, 13, 17), (Color::White, 17, 16),
            (Color::Black, 13, 14), (Color::White, 15, 14), (Color::Black, 13, 13), (Color::White, 14, 12),
            (Color::Black,  5,  2), (Color::White, 11,  9), (Color::Black, 11,  8), (Color::White,  8,  8),
            (Color::Black,  8,  6), (Color::White,  4,  1), (Color::Black,  1, 14), (Color::White,  1, 15),
            (Color::Black,  2, 15), (Color::White,  2, 14), (Color::Black,  2, 16), (Color::White,  1, 16),
            (Color::Black,  3, 14), (Color::White,  2, 13), (Color::Black,  3, 16), (Color::White,  4, 15),
            (Color::Black,  1, 13), (Color::White,  1, 17), (Color::Black,  2, 12), (Color::White,  3, 13),
            (Color::Black,  4, 12), (Color::White,  3, 12), (Color::Black,  3, 11)
        ];

        let mut board = Board::new(7.5);

        for &(color, x, y) in moves.into_iter() {
            board.place(color, x, y);
        }

        assert_eq!(board.inner.is_ladder_escape(Color::White, 251), true);
    }

    // Test that self-atari on the first move is not a ladder.
    #[test]
    fn not_ladder_due_to_self_atari() {
        let moves = [
            (Color::Black,  1,  2), (Color::White,  2,  4), (Color::Black,  2,  3), (Color::White,  1,  5),
            (Color::Black,  1,  4)
        ];

        let mut board = Board::new(7.5);

        for &(color, x, y) in moves.into_iter() {
            board.place(color, x, y);
        }

        assert_eq!(board.inner.is_ladder_capture(Color::White, 3 * 19 + 1), false);  // (Color::White, 1, 3)
    }

    // Test that self-atari of a neighbouring group is not a ladder.
    #[test]
    fn not_ladder_due_to_self_atari_2() {
        let moves = [
            (Color::Black,  3,  4), (Color::White,  2,  4), (Color::Black,  2,  3), (Color::White,  1,  5),
            (Color::Black,  1,  4)
        ];

        let mut board = Board::new(7.5);

        for &(color, x, y) in moves.into_iter() {
            board.place(color, x, y);
        }

        assert_eq!(board.inner.is_ladder_capture(Color::White, 3 * 19 + 1), false);  // (Color::White, 1, 3)
    }
}
