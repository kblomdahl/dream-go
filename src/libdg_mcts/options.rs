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

use dg_go::{Board, Color};
use tree;

pub trait SearchOptions : Clone {
    /// Returns true if the given move should be considered during search.
    ///
    /// # Arguments
    ///
    /// * `board` -
    /// * `to_move` -
    /// * `index` -
    ///
    fn is_policy_candidate(board: &Board, to_move: Color, index: usize) -> bool;

    /// Returns true if the search should be deterministic.
    fn deterministic() -> bool;
}

#[derive(Clone)]
pub struct StandardSearch;

impl SearchOptions for StandardSearch {
    fn is_policy_candidate(_board: &Board, _to_move: Color, _index: usize) -> bool {
        true
    }

    fn deterministic() -> bool {
        false
    }
}

#[derive(Clone)]
pub struct ScoringSearch;

impl SearchOptions for ScoringSearch {
    fn is_policy_candidate(board: &Board, to_move: Color, index: usize) -> bool {
        index != 361 && !is_eye(board, to_move, index)
    }

    fn deterministic() -> bool {
        true
    }
}

/// Returns true if the given vertex is is occupied by a stone of the same color.
///
/// # Arguments
///
/// * `board` -
/// * `color` -
/// * `index` -
/// * `dx` -
/// * `dy` -
///
fn is_vertex_filled(board: &Board, color: Color, index: usize, dx: i8, dy: i8) -> bool {
    let (x, y) = (tree::X[index] as isize, tree::Y[index] as isize);
    let other_x = x + dx as isize;
    let other_y = y + dy as isize;

    other_x >= 0 && other_x < 19 &&
        other_y >= 0 && other_y < 19 &&
        board.at(other_x as usize, other_y as usize) == Some(color)
}

/// Returns true if the given move would fill ones own eye. An eye in this case
/// is recognized as an empty spot that is surrounded by at least 7 stones of
/// the same color. This will miss some _complicated_ eyes, but this is good
/// enough for the heuristic.
///
/// # Arguments
///
/// * `board` -
/// * `color` -
/// * `index` -
///
fn is_eye(board: &Board, color: Color, index: usize) -> bool {
    const CROSS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    const DIAGONAL: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

    let num_cross = CROSS.iter()
        .filter(|(dx, dy)| is_vertex_filled(board, color, index, *dx, *dy))
        .count();
    let num_diagonal = DIAGONAL.iter()
        .filter(|(dx, dy)| is_vertex_filled(board, color, index, *dx, *dy))
        .count();

    // distinguish between the three different cases, (i) an eye in the middle,
    // (ii) an eye in along the edge, and (iii) an eye in the corner.
    let (x, y) = (tree::X[index] as usize, tree::Y[index] as usize);

    if index == 0 || index == 18 || index == 342 || index == 360 {
        num_cross >= 2 && num_diagonal >= 1  // corner move
    } else if x == 0 || x == 18 || y == 0 || y == 18 {
        num_cross >= 3 && num_diagonal >= 2  // edge
    } else {
        num_cross >= 4 && num_diagonal >= 3
    }
}
