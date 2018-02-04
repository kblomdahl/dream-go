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

use go::board_fast::BoardFast;
use go::board::Board;
use go::color::Color;

use std::collections::VecDeque;

pub trait Score {
    fn is_scoreable(&self) -> bool;
    fn get_score(&self) -> (usize, usize);
}

impl Score for Board {
    /// Returns true if this game is fully scorable, a game is
    /// defined as scorable if the following conditions hold:
    ///
    /// * Both black and white has played at least one stone
    /// * All empty vertices are only reachable from one color
    ///
    fn is_scoreable(&self) -> bool {
        let some_black = (0..361).any(|i| self.inner.vertices[i] == Color::Black as u8);
        let some_white = (0..361).any(|i| self.inner.vertices[i] == Color::White as u8);

        some_black && some_white && {
            let black_distance = get_territory_distance(&self.inner, Color::Black);
            let white_distance = get_territory_distance(&self.inner, Color::White);

            (0..361).all(|i| black_distance[i] == 0xff || white_distance[i] == 0xff)
        }
    }

    /// Returns the score for each player `(black, white)` of the
    /// current board state according to the Tromp-Taylor rules.
    ///
    /// This method does not take any komi into account, you will
    /// need to add it yourself.
    fn get_score(&self) -> (usize, usize) {
        let mut black = 0;
        let mut white = 0;

        if self.zobrist_hash != 0 {  // at least one stone has been played
            let black_distance = get_territory_distance(&self.inner, Color::Black);
            let white_distance = get_territory_distance(&self.inner, Color::White);

            for i in 0..361 {
                if black_distance[i] == 0 as u8 {
                    black += 1;  // black has stone at vertex
                } else if white_distance[i] == 0 as u8 {
                    white += 1;  // white has stone at vertex
                } else if white_distance[i] == 0xff {
                    black += 1;  // only reachable from black
                } else if black_distance[i] == 0xff {
                    white += 1;  // only reachable from white
                }
            }
        }

        (black, white)
    }
}

/// Returns an array containing the (manhattan) distance to the closest stone
/// of the given color for each point on the board.
///
/// # Arguments
///
/// * `color` - the color to get the distance from
///
fn get_territory_distance(board: &BoardFast, color: Color) -> [u8; 368] {
    let current = color as u8;

    // find all of our stones and mark them as starting points
    let mut territory = [0xff; 368];
    let mut probes = VecDeque::with_capacity(512);

    for index in 0..361 {
        if board.vertices[index] == current {
            territory[index] = 0;
            probes.push_back(index);
        }
    }

    // compute the distance to all neighbours using a dynamic programming
    // approach where we at each iteration try to update the neighbours of
    // each updated vertex, and if the distance we tried to set was smaller
    // than the current distance we try to update that vertex neighbours.
    //
    // This is equivalent to a Bellman–Ford algorithm for the shortest path.
    while !probes.is_empty() {
        let index = probes.pop_front().unwrap();
        let t = territory[index] + 1;

        foreach_4d!(board, index, |other_index, value| {
            if value == 0 && territory[other_index] > t {
                probes.push_back(other_index);
                territory[other_index] = t;
            }
        });
    }

    territory
}

#[cfg(test)]
mod tests {
    use go::*;

    #[test]
    fn score_black() {
        let mut board = Board::new();
        board.place(Color::Black, 0, 0);

        assert!(!board.is_scoreable());
        assert_eq!(board.get_score(), (361, 0));
    }

    #[test]
    fn score_white() {
        let mut board = Board::new();
        board.place(Color::White, 0, 0);

        assert!(!board.is_scoreable());
        assert_eq!(board.get_score(), (0, 361));
    }

    #[test]
    fn score_black_white() {
        let mut board = Board::new();
        board.place(Color::White, 1, 0);
        board.place(Color::White, 0, 1);
        board.place(Color::White, 1, 1);
        board.place(Color::Black, 2, 0);
        board.place(Color::Black, 2, 1);
        board.place(Color::Black, 0, 2);
        board.place(Color::Black, 1, 2);

        assert!(board.is_scoreable());
        assert_eq!(board.get_score(), (357, 4));
    }
}