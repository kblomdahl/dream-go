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

use go::board_fast::{BoardFast, One, Two, N};
use go::sgf::{Sabaki, SgfCoordinate};
use go::board::Board;
use go::color::Color;

use std::collections::VecDeque;

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
    /// current board state after floating and dead stones has been
    /// removed. The Tromp-Taylor rules are used to determine the
    /// score after clean-up.
    ///
    /// This method does not take any komi into account, you will
    /// need to add it yourself.
    fn get_guess_score(&self) -> (usize, usize, String);
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
        if self.zobrist_hash != 0 {
            // at least one stone has been played
            get_tt_score(&self.inner)
        } else {
            (0, 0)
        }
    }

    fn get_guess_score(&self) -> (usize, usize, String) {
        if self.zobrist_hash != 0 {
            // at least one stone has been played
            let mut board = self.clone();
            let mut sgf = String::new();

            // 1. attempt to make all territory solid by maximizing the number
            //    of liberties for all groups.
            let mut current = match board.last_played {
                Some(color) => color.opposite(),
                _ => Color::Black,
            };
            let mut pass_count = 0;

            while pass_count < 2 {
                if let Some(index) = playout_solid(&board, current) {
                    debug_assert!(board._is_valid(current, index));

                    sgf += &format!(";{}[{}]", current, Sabaki::to_sgf(index % 19, index / 19));

                    board._place(current, index);
                    pass_count = 0;
                } else {
                    pass_count += 1;

                    sgf += &format!(";{}[]", current);
                }

                current = current.opposite();
            }

            // 2. attempt to capture any of the opponents groups (safely)
            let mut pass_count = 0;

            while pass_count < 2 {
                if let Some(index) = playout_race(&board, current) {
                    debug_assert!(board._is_valid(current, index));

                    sgf += &format!(";{}[{}]", current, Sabaki::to_sgf(index % 19, index / 19));

                    board._place(current, index);
                    pass_count = 0;
                } else {
                    pass_count += 1;

                    sgf += &format!(";{}[]", current);
                }

                current = current.opposite();
            }

            // 3. remove _obviously_ dead stones from the board
            let mut visited = [false; 361];

            for index in 0..361 {
                for &other_index in get_dead_groups(&board.inner, index, &mut visited).iter() {
                    let color = board.inner.vertices[other_index] as usize;

                    board.inner.capture(color, other_index);
                }
            }

            // do not score the modified board directly, since dame filling
            // occured randomly and those are worth points. Instead use it to
            // determine what stones are dead.
            let black_distance = get_territory_distance(&board.inner, Color::Black);
            let white_distance = get_territory_distance(&board.inner, Color::White);
            let mut other = self.inner.clone();

            for i in 0..361 {
                if other.vertices[i] == board.inner.vertices[i] {
                    // pass
                } else if other.vertices[i] != 0 {
                    if board.inner.vertices[i] == 0 {
                        if other.vertices[i] == Color::Black as u8 && white_distance[i] != 0xff {
                            other.vertices[i] = 0; // black stone is not black territory, nor stone
                        } else if other.vertices[i] == Color::White as u8
                            && black_distance[i] != 0xff
                        {
                            other.vertices[i] = 0; // white stone is not white territory, nor stone
                        }
                    } else {
                        other.vertices[i] = 0; // remove dead stone
                    }
                }
            }

            let (black, white) = get_tt_score(&other);

            (black, white, sgf)
        } else {
            (0, 0, String::new())
        }
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

/// Returns the number of stones that are strongly connected to the given
/// vertex, including any stone on the vertex itself.
///
/// # Arguments
///
/// * `board` -
/// * `index` -
///
fn get_group_size(board: &BoardFast, index: usize) -> usize {
    let mut current = index;
    let mut count = 0;

    loop {
        count += 1;

        current = board.next_vertex[current] as usize;
        if current == index {
            break;
        }
    }

    count
}

/// Returns the number of liberties that the given group will have *if* a stone
/// was played at the given index on the given board.
///
/// # Arguments
///
/// * `board` - the board to play the move on
/// * `color` - the color of the move to play
/// * `index` - the index of the move to play
/// * `group_index` - the group to return the number of liberties of
///
fn get_num_liberties_if(
    board: &BoardFast,
    color: Color,
    index: usize,
    group_index: usize,
) -> usize {
    let mut other = board.clone();

    if other.is_valid(color, index) {
        other.place(color, index);
    }

    if other.vertices[group_index] == 0 {
        0
    } else {
        other.get_n_liberty::<N>(group_index, 361).len()
    }
}

/// Returns the best capture of a neighbouring opponent group.
///
/// # Arguments
///
/// * `board` - the board to capture on
/// * `color` - the color of the player to capture
/// * `index` - the index whose neighbours should be captured
///
fn get_capture_neighbour_move(board: &Board, color: Color, index: usize) -> Option<usize> {
    let opponent = color.opposite();
    let inner = &board.inner;

    // check if we can capture any of the neighbouring groups
    let mut candidates = vec![];
    let mut current = index;

    loop {
        foreach_4d!(inner, current, |other_index, other| {
            if other == opponent as u8 && !inner.has_n_liberty::<Two>(other_index, 2) {
                let liberty = inner.get_n_liberty::<One>(other_index, 1);
                let size = get_group_size(inner, other_index);

                if board._is_valid(color, liberty) {
                    // check if playing this move actually increase our
                    // liberty count
                    if get_num_liberties_if(inner, color, liberty, current) > 1 {
                        candidates.push((size, liberty))
                    }
                }
            }
        });

        current = inner.next_vertex[current] as usize;
        if current == index {
            break;
        }
    }

    // if there are multiple opponent groups to capture, then we want to
    // capture the largest one
    candidates.sort_unstable();

    if !candidates.is_empty() {
        let cand = candidates.last().unwrap();

        Some(cand.1)
    } else {
        None
    }
}

/// Returns the best (if one is available) move that prevents the given group
/// from being captured.
///
/// # Arguments
///
/// * `board` - the board to escape on
/// * `color` - the color of the group at the given index
/// * `index` - the index of the group to escape with
///
fn get_escape_move(board: &Board, color: Color, index: usize) -> Option<usize> {
    let liberty = board.inner.get_n_liberty::<One>(index, 1);

    if board._is_valid(color, liberty)
        && get_num_liberties_if(&board.inner, color, liberty, index) > 1
    {
        Some(liberty)
    } else {
        None
    }
}

/// Returns if the given groups is in seki with one of its neighbouring groups.
///
/// # Arguments
///
/// * `board` -
/// * `color` - the color of the group at `index`
/// * `index` - the group to check for seki
///
fn is_seki(board: &BoardFast, color: Color, index: usize) -> bool {
    fn _is_seki(board: &BoardFast, index: usize, other: usize) -> bool {
        let group_liberties = board.get_n_liberty::<N>(index, 361);
        let other_liberties = board.get_n_liberty::<N>(other, 361);

        group_liberties.len() == other_liberties.len() && {
            group_liberties.into_iter()
                .all(|liberty| other_liberties.contains(&liberty)) && {
                let group_size = get_group_size(&board, index);
                let other_size = get_group_size(&board, other);

                group_size > 3 && other_size > 3
            }
        }
    }

    let liberties = board.get_n_liberty::<N>(index, 361);
    let mut visited = [false; 361];

    for liberty in liberties {
        foreach_4d!(board, liberty, |other_index, other| {
            if other == color.opposite() as u8 && !visited[other_index] {
                if _is_seki(board, index, other_index) {
                    return true;
                }

                // mark all strongly connected stones as visited
                let mut current = other_index;

                loop {
                    visited[current] = true;

                    current = board.next_vertex[current] as usize;
                    if current == other_index {
                        break;
                    }
                }
            }
        });
    }

    false
}

/// Returns if the given groups is in a semeai with one of its neighbouring
/// groups.
///
/// # Arguments
///
/// * `board` -
/// * `color` - the color of the group at `index`
/// * `index` - the group to check for semeai
///
fn is_semeai(board: &BoardFast, color: Color, index: usize) -> Option<usize> {
    fn _is_semeai(board: &BoardFast, color: Color, index: usize, other: usize) -> bool {
        let group_liberties = board.get_n_liberty::<N>(index, 361);
        let other_liberties = board.get_n_liberty::<N>(other, 361);

        group_liberties.len() == other_liberties.len() && {
            // check if we can possibly kill the opponent, or if at least two
            // of the liberties can never be filled
            let eyes = other_liberties.into_iter()
                .filter(|&liberty| !board.is_valid(color, liberty))
                .count();

            eyes < 2
        }
    }

    // check if this group is alive, if it is then it is not in a semeai with
    // anything
    let liberties = board.get_n_liberty::<N>(index, 361);
    let eyes = liberties.iter()
        .filter(|&&liberty| !board.is_valid(color.opposite(), liberty))
        .count();

    if eyes >= 2 {
        return None;
    }

    // check if it has the same number of liberties as any neighbouring
    // group (that does not have two eyes)
    let mut min_neighbour = ::std::usize::MAX;
    let mut candidates = vec! [];
    let mut visited = [false; 361];

    for liberty in liberties {
        foreach_4d!(board, liberty, |other_index, other| {
            if other == color.opposite() as u8 && !visited[other_index] {
                if _is_semeai(board, color, index, other_index) {
                    let liberty_count = board.get_n_liberty::<N>(other_index, 361).len();

                    candidates.push((other_index, liberty_count));
                } else {
                    min_neighbour = ::std::cmp::min(
                        board.get_n_liberty::<N>(other_index, 361).len(),
                        min_neighbour
                    );
                }

                // mark all strongly connected stones as visited
                let mut current = other_index;

                loop {
                    visited[current] = true;

                    current = board.next_vertex[current] as usize;
                    if current == other_index {
                        break;
                    }
                }
            }
        });
    }

    // reject any semeais that can be solved by capturing some other group
    candidates.into_iter()
        .filter(|&(_index, liberty_count)| liberty_count >= min_neighbour)
        .map(|(index, _liberty_count)| index)
        .next()
}

/// Returns a move that might be able to capture the given group in a
/// semeai (capturing race).
///
/// # Arguments
///
/// * `board` - the board to capture on
/// * `color` - the color of the stone to place on the board
/// * `index` - the index of the group to attempt to capture in a semeai
///
fn get_capture_semeai_moves(board: &Board, color: Color, index: usize) -> Vec<usize> {
    let inner = &board.inner;

    inner.get_n_liberty::<N>(index, 361).into_iter()
        .filter(|&other_liberty| board._is_valid(color, other_liberty))
        .filter(|&other_liberty| {
            let cand_liberties = get_num_liberties_if(inner, color, other_liberty, other_liberty);
            let group_liberties = get_num_liberties_if(inner, color, other_liberty, index);

            cand_liberties >= group_liberties
        })
        .collect::<Vec<_>>()
}

/// Returns the number of liberties the entire board will gain from capturing
/// the group at `other_index`. The given `color` and `index` should be the
/// move that captures the group at `other_index`.
///
/// # Arguments
///
/// * `board` - the board to capture on
/// * `color` - the color of the capturing move
/// * `index` - the capturing move
/// * `other_index` - index of the group to capture
///
fn get_all_liberties(board: &BoardFast, color: Color, index: usize, other_index: usize) -> usize {
    let mut other = board.clone();
    other.place(color, index);

    // iterate over each neighbouring group and count the number of liberties
    // before and after the capture
    let mut current = other_index;
    let mut visited = [false; 361];
    let mut total = 0isize;

    loop {
        foreach_4d!(other, current, |group_index, group| {
            if group == color as u8 && !visited[group_index] {
                let pre = if board.vertices[group_index] != 0 {
                    board.get_n_liberty::<N>(group_index, 361).len() as isize
                } else {
                    0isize
                };
                let post = other.get_n_liberty::<N>(group_index, 361).len() as isize;

                total += post - pre;

                // mark as strongly connected vertices as visited
                let mut current = group_index;

                loop {
                    visited[current] = true;

                    current = other.next_vertex[current] as usize;
                    if current == group_index {
                        break;
                    }
                }
            }
        });

        current = board.next_vertex[current] as usize;
        if current == other_index {
            break;
        }
    }

    if total < 0 {
        0
    } else {
        total as usize
    }
}

/// Find the move for the given `color` and `board` that increase the number of
/// liberties the available most.
///
/// # Arguments
///
/// * `board` - the board to find a move on
/// * `color` - the color to find a move for
///
fn playout_solid(board: &Board, color: Color) -> Option<usize> {
    let inner = &board.inner;

    let mut candidates = vec![];
    let mut visited = [false; 361];

    for i in 0..361 {
        if !visited[i] && inner.vertices[i] == color as u8 {
            let liberties = inner.get_n_liberty::<N>(i, 361);
            let num_liberties = liberties.len();
            let liberties = liberties.into_iter()
                .filter(|&liberty| board._is_valid(color, liberty));

            for liberty in liberties {
                let post = get_num_liberties_if(inner, color, liberty, liberty);

                // check so that at least one group this liberty connects to
                // gain from the connection, and that none of the groups lose
                // anything.
                let mut non_decrease = true;
                let mut some_increase = false;
                let mut min_pre = ::std::usize::MAX;

                foreach_4d!(inner, liberty, |other_index, other| {
                    if other == color as u8 {
                        let pre = inner.get_n_liberty::<N>(other_index, 361).len();

                        min_pre = ::std::cmp::min(pre, min_pre);
                        non_decrease = non_decrease && post >= pre;
                        some_increase = some_increase || post > pre;
                    }
                });

                if non_decrease && some_increase {
                    candidates.push((0, post - min_pre, liberty));
                } else if some_increase {
                    let regret = get_num_liberties_if(inner, color.opposite(), liberty, i);

                    // allow net-loss moves if it saves a group from atari
                    if regret == 0 {
                        let group_size = get_group_size(inner, i);

                        candidates.push((group_size, post, liberty));
                    }
                } else if non_decrease {
                    let regret = get_num_liberties_if(inner, color.opposite(), liberty, i);

                    if regret < num_liberties {
                        candidates.push((0, post - regret, liberty));
                    }
                }
            }
        } else if !visited[i] && inner.vertices[i] == color.opposite() as u8 {
            // capture opponent groups that are in atari, but do not attemp to
            // chase them
            if !inner.has_n_liberty::<Two>(i, 2) {
                let liberty = inner.get_n_liberty::<One>(i, 1);

                if board._is_valid(color, liberty) {
                    let post = get_all_liberties(inner, color, liberty, i);
                    let group_size = get_group_size(inner, i);

                    candidates.push((group_size, post, liberty));
                }
            }
        }

        if !visited[i] && inner.vertices[i] != 0 {
            // mark all strongly connected stones as visited
            let mut current = i;

            loop {
                visited[current] = true;

                current = inner.next_vertex[current] as usize;
                if current == i {
                    break;
                }
            }
        }
    }

    candidates.sort_unstable_by_key(|&(points, liberty_gain, _index)| (points, liberty_gain));
    candidates.last().map(|&(_liberty_gain, _gain, index)| index)
}

/// Find a move for the given `board` and `color` that best captures or escapes
/// capture of an opponent group.
///
/// # Arguments
///
/// * `board` - the board to find a move on
/// * `color` - the color to find a move for
///
fn playout_race(board: &Board, color: Color) -> Option<usize> {
    let inner = &board.inner;

    let mut candidates = vec![];
    let mut visited = [false; 361];

    for i in 0..361 {
        if !visited[i] && inner.vertices[i] == color as u8 {
            if !inner.has_n_liberty::<Two>(i, 2) {
                let group_size = get_group_size(inner, i);

                // this group is in atari, we can handle this by:
                // 1. counter-capture some of the opponents groups to gain
                //    more liberties
                if let Some(capture) = get_capture_neighbour_move(board, color, i) {
                    let liberty_gain = get_num_liberties_if(inner, color, capture, i) - 1;

                    candidates.push((group_size, liberty_gain, capture));
                }
                // 2. extend into our single liberty
                else if let Some(escape) = get_escape_move(board, color, i) {
                    let liberty_gain = get_num_liberties_if(inner, color, escape, i) - 1;

                    candidates.push((group_size, liberty_gain, escape));
                }
            }
        } else if !visited[i] && inner.vertices[i] == color.opposite() as u8 {
            if is_seki(inner, color.opposite(), i) {
                // pass
            } else {
                let num_liberties = inner.get_n_liberty::<N>(i, 361).len();
                let group_size = if let Some(other) = is_semeai(inner, color.opposite(), i) {
                    get_group_size(inner, other)
                } else {
                    0
                };

                let is_capture = num_liberties == 1;
                let is_throwin = num_liberties == 2;

                for capture in get_capture_semeai_moves(board, color, i) {
                    let liberty_gain = get_num_liberties_if(inner, color, capture, capture);

                    if is_capture || (liberty_gain > 1 || is_throwin) {
                        // don't self-atari
                        candidates.push((group_size, liberty_gain, capture));
                    }
                }
            }
        }

        // mark all strongly connected stones as visited
        if inner.vertices[i] != 0 {
            let mut current = i;

            loop {
                visited[current] = true;

                current = inner.next_vertex[current] as usize;
                if current == i {
                    break;
                }
            }
        }
    }

    candidates.sort_unstable_by_key(|&(gain, liberty_gain, _index)| (gain, liberty_gain));

    candidates
        .last()
        .map(|&(_gain, _liberty_gain, index)| index)
}

/// Return an array containing all stones that can be reached from the given
/// index and can be determined dead according to some simplistic static
/// analysis.
///
/// # Arguments
///
/// * `board` - the board to search for dead groups on
/// * `index` - the starting point
/// * `visited` - memoizing array to avoid visiting the same vertex twice
///
fn get_dead_groups(board: &BoardFast, index: usize, visited: &mut [bool]) -> Vec<usize> {
    const WHITE: u8 = Color::White as u8;
    const BLACK: u8 = Color::Black as u8;

    if visited[index] || board.vertices[index] != 0 {
        vec![]
    } else {
        let black_area = get_reachable(board, index, |value| value == 0 || value == BLACK);
        let white_area = get_reachable(board, index, |value| value == 0 || value == WHITE);

        let black_vertices = black_area.iter().cloned()
            .filter(|&index| board.vertices[index] == BLACK)
            .collect::<Vec<_>>();
        let white_vertices = white_area.iter().cloned()
            .filter(|&index| board.vertices[index] == WHITE)
            .collect::<Vec<_>>();

        // determine the eye space of any group that we reached, this can be
        // done by checking area was only reached during the black (or white)
        // probe while excluding vertices.
        let black_eyes = black_area.iter().cloned()
            .filter(|index| {
                let isnt_vertex = black_vertices.binary_search(index).is_err();
                let isnt_white = white_area.binary_search(index).is_err();

                isnt_vertex && isnt_white
            })
            .collect::<Vec<_>>();

        let white_eyes = white_area.iter().cloned()
            .filter(|index| {
                let isnt_vertex = white_vertices.binary_search(index).is_err();
                let isnt_black = black_area.binary_search(index).is_err();

                isnt_vertex && isnt_black
            })
            .collect::<Vec<_>>();

        // static analysis to determine whether a group is alive or not:
        //
        // - if a group only has at most one eye then it is in semeai.
        // - if a group in semeai is only reachable by groups in semeai (with
        //   the same number of stones) then it is in seki.
        // - if a group in semeai has less eyes or stones than its competing
        //   group then it is dead.
        //
        // this assumes that the human(s) has already done most of the heavy
        // lifting and reduced the number of eyes and liberties, etc, of the
        // groups. So should work pretty well as a final score measure, but not
        // in the middle of the game.
        //
        let num_blacks = black_vertices.len();
        let num_whites = white_vertices.len();
        let mut result = vec![];

        if num_blacks == num_whites {
            if black_eyes.len() == 1 && white_eyes.len() == 1 {
                // this might be a seki, better leave it alone
                let area = get_reachable(board, index, |value| value == 0);

                for other_index in area.into_iter() {
                    visited[other_index] = true;
                }
            } else if black_eyes.len() <= 1 && white_eyes.len() >= 1 {
                // black is dead
                for other_index in black_area.into_iter() {
                    visited[other_index] = true;
                }

                result = black_vertices;
            } else if white_eyes.len() <= 1 && black_eyes.len() >= 1 {
                // white is dead
                for other_index in white_area.into_iter() {
                    visited[other_index] = true;
                }

                result = white_vertices;
            }
        } else if black_eyes.len() <= 1 && num_blacks < num_whites {
            // black is dead in a semeai (or worse)
            for other_index in black_area.into_iter() {
                visited[other_index] = true;
            }

            result = black_vertices;
        } else if white_eyes.len() <= 1 && num_whites < num_blacks {
            // white is dead in a semeai (or worse)
            for other_index in white_area.into_iter() {
                visited[other_index] = true;
            }

            result = white_vertices;
        }

        result
    }
}

/// Returns a sorted vector containing all vertices on the given board that can
/// be reached from the given index by traversing only vertices that fulfill
/// the given predicate.
///
/// # Arguments
///
/// * `board` - the board to traverse
/// * `index` - the starting point
/// * `pred` - the predicate that a vertex must fulfill to be allowed to be
///   traversed
///
fn get_reachable<P: FnMut(u8) -> bool>(board: &BoardFast, index: usize, mut pred: P) -> Vec<usize> {
    debug_assert!(pred(board.vertices[index]));

    let mut visited = [false; 361];
    let mut reachable = vec![];
    let mut candidates = VecDeque::with_capacity(512);

    candidates.push_back(index);
    visited[index] = true;

    while !candidates.is_empty() {
        let cand = candidates.pop_front().unwrap();

        foreach_4d!(board, cand, |next_cand, value| {
            if pred(value) && !visited[next_cand] {
                candidates.push_back(next_cand);
                visited[next_cand] = true;
            }
        });

        reachable.push(cand);
    }

    reachable.sort_unstable();
    reachable
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

    #[test]
    fn guess_score() {
        // Yun Chanhee (7p) vs Hong Seongji (9p) -- 23rd Korean GS Caltex Cup
        let moves = [
            (Color::Black, 15, 3),
            (Color::White, 3, 3),
            (Color::Black, 16, 15),
            (Color::White, 3, 15),
            (Color::Black, 13, 15),
            (Color::White, 5, 16),
            (Color::Black, 6, 15),
            (Color::White, 11, 16),
            (Color::Black, 2, 2),
            (Color::White, 2, 3),
            (Color::Black, 3, 2),
            (Color::White, 4, 3),
            (Color::Black, 5, 1),
            (Color::White, 2, 9),
            (Color::Black, 3, 12),
            (Color::White, 2, 14),
            (Color::Black, 4, 9),
            (Color::White, 4, 10),
            (Color::Black, 5, 10),
            (Color::White, 3, 10),
            (Color::Black, 5, 11),
            (Color::White, 5, 9),
            (Color::Black, 4, 8),
            (Color::White, 6, 9),
            (Color::Black, 2, 7),
            (Color::White, 2, 12),
            (Color::Black, 6, 7),
            (Color::White, 8, 9),
            (Color::Black, 7, 10),
            (Color::White, 7, 9),
            (Color::Black, 4, 6),
            (Color::White, 8, 12),
            (Color::Black, 6, 13),
            (Color::White, 8, 14),
            (Color::Black, 6, 16),
            (Color::White, 5, 15),
            (Color::Black, 5, 14),
            (Color::White, 7, 14),
            (Color::Black, 6, 14),
            (Color::White, 5, 17),
            (Color::Black, 8, 16),
            (Color::White, 10, 14),
            (Color::Black, 9, 15),
            (Color::White, 9, 14),
            (Color::Black, 2, 13),
            (Color::White, 2, 11),
            (Color::Black, 4, 15),
            (Color::White, 3, 16),
            (Color::Black, 9, 11),
            (Color::White, 8, 11),
            (Color::Black, 11, 15),
            (Color::White, 10, 15),
            (Color::Black, 3, 13),
            (Color::White, 1, 13),
            (Color::Black, 10, 16),
            (Color::White, 11, 17),
            (Color::Black, 10, 17),
            (Color::White, 15, 15),
            (Color::Black, 8, 10),
            (Color::White, 9, 10),
            (Color::Black, 10, 11),
            (Color::White, 12, 13),
            (Color::Black, 12, 15),
            (Color::White, 16, 16),
            (Color::Black, 9, 9),
            (Color::White, 10, 10),
            (Color::Black, 1, 14),
            (Color::White, 1, 15),
            (Color::Black, 3, 14),
            (Color::White, 0, 14),
            (Color::Black, 11, 11),
            (Color::White, 9, 8),
            (Color::Black, 10, 9),
            (Color::White, 11, 10),
            (Color::Black, 13, 13),
            (Color::White, 6, 10),
            (Color::Black, 7, 11),
            (Color::White, 12, 10),
            (Color::Black, 12, 12),
            (Color::White, 13, 12),
            (Color::Black, 11, 13),
            (Color::White, 6, 5),
            (Color::Black, 5, 4),
            (Color::White, 5, 2),
            (Color::Black, 6, 4),
            (Color::White, 6, 1),
            (Color::Black, 4, 1),
            (Color::White, 7, 2),
            (Color::Black, 7, 5),
            (Color::White, 14, 13),
            (Color::Black, 12, 14),
            (Color::White, 10, 8),
            (Color::Black, 16, 14),
            (Color::White, 15, 16),
            (Color::Black, 16, 11),
            (Color::White, 13, 17),
            (Color::Black, 13, 11),
            (Color::White, 15, 14),
            (Color::Black, 14, 12),
            (Color::White, 16, 2),
            (Color::Black, 16, 3),
            (Color::White, 15, 2),
            (Color::Black, 14, 2),
            (Color::White, 14, 1),
            (Color::Black, 13, 2),
            (Color::White, 13, 1),
            (Color::Black, 12, 2),
            (Color::White, 6, 11),
            (Color::Black, 7, 12),
            (Color::White, 14, 4),
            (Color::Black, 9, 3),
            (Color::White, 7, 6),
            (Color::Black, 8, 6),
            (Color::White, 14, 3),
            (Color::Black, 17, 3),
            (Color::White, 11, 2),
            (Color::Black, 12, 1),
            (Color::White, 17, 2),
            (Color::Black, 13, 0),
            (Color::White, 15, 0),
            (Color::Black, 9, 2),
            (Color::White, 1, 2),
            (Color::Black, 1, 1),
            (Color::White, 8, 4),
            (Color::Black, 8, 5),
            (Color::White, 9, 4),
            (Color::Black, 10, 4),
            (Color::White, 10, 5),
            (Color::Black, 9, 5),
            (Color::White, 11, 4),
            (Color::Black, 10, 3),
            (Color::White, 2, 5),
            (Color::Black, 1, 6),
            (Color::White, 1, 5),
            (Color::Black, 1, 3),
            (Color::White, 1, 4),
            (Color::Black, 0, 2),
            (Color::White, 2, 6),
            (Color::Black, 10, 6),
            (Color::White, 16, 5),
            (Color::Black, 11, 5),
            (Color::White, 7, 7),
            (Color::Black, 8, 7),
            (Color::White, 6, 12),
            (Color::Black, 7, 13),
            (Color::White, 5, 7),
            (Color::Black, 4, 7),
            (Color::White, 6, 6),
            (Color::Black, 4, 4),
            (Color::White, 2, 8),
            (Color::Black, 17, 5),
            (Color::White, 16, 6),
            (Color::Black, 17, 6),
            (Color::White, 15, 4),
            (Color::Black, 12, 8),
            (Color::White, 12, 7),
            (Color::Black, 16, 7),
            (Color::White, 17, 4),
            (Color::Black, 13, 8),
            (Color::White, 11, 8),
            (Color::Black, 17, 8),
            (Color::White, 12, 4),
            (Color::Black, 11, 3),
            (Color::White, 7, 3),
            (Color::Black, 7, 4),
            (Color::White, 18, 4),
            (Color::Black, 17, 16),
            (Color::White, 16, 13),
            (Color::Black, 17, 13),
            (Color::White, 17, 17),
            (Color::Black, 8, 8),
            (Color::White, 5, 12),
            (Color::Black, 4, 11),
            (Color::White, 11, 7),
            (Color::Black, 13, 6),
            (Color::White, 13, 7),
            (Color::Black, 14, 7),
            (Color::White, 12, 5),
            (Color::Black, 12, 6),
            (Color::White, 11, 6),
            (Color::Black, 14, 6),
            (Color::White, 10, 5),
            (Color::Black, 3, 4),
            (Color::White, 2, 4),
            (Color::Black, 11, 5),
            (Color::White, 4, 5),
            (Color::Black, 5, 5),
            (Color::White, 10, 5),
            (Color::Black, 1, 7),
            (Color::White, 3, 7),
            (Color::Black, 11, 5),
            (Color::White, 14, 8),
            (Color::Black, 14, 9),
            (Color::White, 10, 5),
            (Color::Black, 3, 6),
            (Color::White, 3, 8),
            (Color::Black, 11, 5),
            (Color::White, 15, 8),
            (Color::Black, 13, 9),
            (Color::White, 10, 5),
            (Color::Black, 4, 2),
            (Color::White, 5, 3),
            (Color::Black, 11, 5),
            (Color::White, 15, 7),
            (Color::Black, 15, 6),
            (Color::White, 10, 5),
            (Color::Black, 6, 3),
            (Color::White, 6, 2),
            (Color::Black, 11, 5),
            (Color::White, 4, 16),
            (Color::Black, 4, 14),
            (Color::White, 10, 5),
            (Color::Black, 8, 3),
            (Color::White, 7, 17),
            (Color::Black, 7, 16),
            (Color::White, 11, 5),
            (Color::Black, 17, 15),
            (Color::White, 18, 17),
            (Color::Black, 8, 1),
            (Color::White, 8, 17),
            (Color::Black, 6, 17),
            (Color::White, 6, 18),
            (Color::Black, 9, 18),
            (Color::White, 7, 18),
            (Color::Black, 7, 8),
            (Color::White, 9, 17),
            (Color::Black, 9, 16),
            (Color::White, 11, 18),
            (Color::Black, 8, 13),
            (Color::White, 6, 8),
            (Color::Black, 16, 12),
            (Color::White, 15, 13),
            (Color::Black, 18, 16),
            (Color::White, 18, 6),
            (Color::Black, 17, 7),
            (Color::White, 18, 5),
            (Color::Black, 15, 9),
            (Color::White, 12, 3),
            (Color::Black, 11, 1),
            (Color::White, 10, 18),
            (Color::Black, 13, 16),
            (Color::White, 12, 17),
            (Color::Black, 6, 0),
            (Color::White, 7, 0),
            (Color::Black, 5, 0),
            (Color::White, 7, 1),
            (Color::Black, 14, 16),
            (Color::White, 14, 17),
            (Color::Black, 0, 4),
            (Color::White, 1, 8),
            (Color::Black, 8, 0),
            (Color::White, 18, 7),
            (Color::Black, 18, 8),
            (Color::White, 13, 10),
            (Color::Black, 14, 10),
            (Color::White, 15, 12),
            (Color::Black, 15, 11),
            (Color::White, 0, 7),
            (Color::Black, 3, 5),
            (Color::White, 12, 11),
            (Color::Black, 8, 2),
            (Color::White, 0, 6),
            (Color::Black, 9, 12),
            (Color::White, 11, 9),
            (Color::Black, 9, 9),
            (Color::White, 13, 3),
            (Color::Black, 13, 5),
            (Color::White, 9, 7),
            (Color::Black, 9, 6),
            (Color::White, 14, 0),
            (Color::Black, 12, 0),
            (Color::White, 16, 4),
            (Color::Black, 5, 6),
            (Color::White, 6, 7),
            (Color::Black, 4, 12),
            (Color::White, 5, 13),
            (Color::Black, 3, 9),
            (Color::White, 3, 11),
            (Color::Black, 16, 8),
            (Color::White, 18, 3),
            (Color::Black, 5, 8),
            (Color::White, 10, 9),
            (Color::Black, 14, 15),
            (Color::White, 14, 14),
            (Color::Black, 4, 13),
            (Color::White, 9, 9),
            (Color::Black, 13, 14),
            (Color::White, 0, 5),
            (Color::Black, 0, 3),
            (Color::White, 10, 7),
            (Color::Black, 12, 9),
            (Color::White, 13, 4),
            (Color::Black, 14, 5),
            (Color::White, 15, 5),
            (Color::Black, 12, 16),
        ];

        let mut board = Board::new();

        for &(color, x, y) in moves.into_iter() {
            board.place(color, x, y);
        }

        let (black, white, _rollout) = board.get_guess_score();

        assert_eq!((black, white), (184, 177));
    }
}
