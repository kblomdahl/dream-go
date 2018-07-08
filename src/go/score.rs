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

#[derive(Debug, PartialEq, Eq)]
pub enum StoneStatus {
    Alive,
    Dead,
    Seki
}

impl ::std::str::FromStr for StoneStatus {
    type Err = ();

    fn from_str(s: &str) -> Result<StoneStatus, Self::Err> {
        let s = s.to_lowercase();

        if s == "alive" {
            Ok(StoneStatus::Alive)
        } else if s == "dead" {
            Ok(StoneStatus::Dead)
        } else if s == "seki" {
            Ok(StoneStatus::Seki)
        } else {
            Err(())
        }
    }
}

pub trait Score {
    /// Returns true if this game is fully scorable, a game is
    /// defined as scorable if the following conditions hold:
    ///
    /// * Both black and white has played at least one stone
    /// * All empty vertices are only reachable from one color
    ///
    fn is_scoreable(&self) -> bool;

    /// Returns the score for each player `(black, white)` of the
    /// current board state according to the Tromp-Taylor rules.
    ///
    /// This method does not take any komi into account, you will
    /// need to add it yourself.
    fn get_score(&self) -> (usize, usize);

    /// Returns the score for each player `(black, white)` of the
    /// current board state after any stones that are not part of
    /// the given _finished_ board state. The Tromp-Taylor rules are
    /// used to determine the score after clean-up.
    ///
    /// This method does not take any komi into account, you will
    /// need to add it yourself.
    /// 
    /// # Arguments
    /// 
    /// * `finished` - A copy of this board that has been played to
    ///   finish, using some heuristic
    /// 
    fn get_guess_score(&self, finished: &Board) -> (usize, usize);

    /// Returns the status of all stones on the board:
    /// 
    /// - **alive** if the stone is present on both
    /// - **dead** if the stone is not present in the _finished_ board
    /// - **seki** if the stone is present on both, but not scoreable
    /// 
    /// # Arguments
    /// 
    /// * `finished` - A copy of this board that has been played to
    ///   finish, using some heuristic
    fn get_stone_status(&self, finished: &Board) -> Vec<(usize, StoneStatus)>;
}

impl Score for Board {
    fn is_scoreable(&self) -> bool {
        let some_black = (0..361).any(|i| self.inner.vertices[i] == Color::Black as u8);
        let some_white = (0..361).any(|i| self.inner.vertices[i] == Color::White as u8);

        some_black && some_white && {
            let black_distance = get_territory_distance(&self.inner, Color::Black);
            let white_distance = get_territory_distance(&self.inner, Color::White);

            (0..361).all(|i| black_distance[i] == 0xff || white_distance[i] == 0xff)
        }
    }

    fn get_score(&self) -> (usize, usize) {
        if self.zobrist_hash != 0 {  // at least one stone has been played
            get_tt_score(&self.inner)
        } else {
            (0, 0)
        }
    }

    fn get_guess_score(&self, finished: &Board) -> (usize, usize) {
        // do not score the finished board directly, since there might be dame
        // fillings, etc, that we do not want to take into account.
        let black_distance = get_territory_distance(&finished.inner, Color::Black);
        let white_distance = get_territory_distance(&finished.inner, Color::White);
        let mut other = self.inner.clone();

        for i in 0..361 {
            if other.vertices[i] == finished.inner.vertices[i] {
                // pass
            } else if other.vertices[i] != 0 {
                if finished.inner.vertices[i] == 0 {
                    if other.vertices[i] == Color::Black as u8 && white_distance[i] != 0xff {
                        other.vertices[i] = 0; // black stone is not black territory, nor stone
                    } else if other.vertices[i] == Color::White as u8 && black_distance[i] != 0xff {
                        other.vertices[i] = 0; // white stone is not white territory, nor stone
                    }
                } else {
                    other.vertices[i] = 0; // remove dead stone
                }
            }
        }

        get_tt_score(&other)
    }

    fn get_stone_status(&self, finished: &Board) -> Vec<(usize, StoneStatus)> {
        let black_distance = get_territory_distance(&finished.inner, Color::Black);
        let white_distance = get_territory_distance(&finished.inner, Color::White);
        let mut status_list = vec! [];

        for i in 0..361 {
            if self.inner.vertices[i] == finished.inner.vertices[i] {
                if self.inner.vertices[i] != 0 {
                    status_list.push((i, StoneStatus::Alive));
                }
            } else if self.inner.vertices[i] != 0 {
                if finished.inner.vertices[i] == 0 {
                    if self.inner.vertices[i] == Color::Black as u8 && white_distance[i] != 0xff {
                        status_list.push((i, StoneStatus::Dead));
                    } else if self.inner.vertices[i] == Color::White as u8 && black_distance[i] != 0xff {
                        status_list.push((i, StoneStatus::Dead));
                    }
                } else {
                    status_list.push((i, StoneStatus::Dead));
                }
            }
        }

        status_list
    }
}

/// Returns the score of the given board according to the Tromp-Taylor
/// rules.
///
/// # Arguments
///
/// * `board` - the board to score
///
fn get_tt_score(board: &BoardFast) -> (usize, usize) {
    let mut black = 0;
    let mut white = 0;
    let black_distance = get_territory_distance(&board, Color::Black);
    let white_distance = get_territory_distance(&board, Color::White);

    for i in 0..361 {
        if black_distance[i] == 0 as u8 {
            black += 1; // black has stone at vertex
        } else if white_distance[i] == 0 as u8 {
            white += 1; // white has stone at vertex
        } else if white_distance[i] == 0xff {
            black += 1; // only reachable from black
        } else if black_distance[i] == 0xff {
            white += 1; // only reachable from white
        }
    }

    (black, white)
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
        let mut board = Board::new(DEFAULT_KOMI);
        board.place(Color::Black, 0, 0);

        assert!(!board.is_scoreable());
        assert_eq!(board.get_score(), (361, 0));
    }

    #[test]
    fn score_white() {
        let mut board = Board::new(DEFAULT_KOMI);
        board.place(Color::White, 0, 0);

        assert!(!board.is_scoreable());
        assert_eq!(board.get_score(), (0, 361));
    }

    #[test]
    fn score_black_white() {
        let mut board = Board::new(DEFAULT_KOMI);
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
