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

use ordered_float::OrderedFloat;

use go::{Board, Color};
use mcts::*;

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
    const DELTA: [i8; 8] = [-20, -19, -18, -1, 1, 18, 19, 20];

    let count = DELTA.iter()
        .map(|d| index as isize + *d as isize)
        .filter(|other| *other >= 0 && *other < 361)
        .filter(|other| {
            let other = *other as usize;
            let (x, y) = (tree::X[other] as usize, tree::Y[other] as usize);

            board.at(x, y) == Some(color)
        })
        .count();

    // distinguish between the three different cases, (i) an eye in the middle,
    // (ii) an eye in along the edge, and (iii) an eye in the corner.
    let (x, y) = (tree::X[index] as usize, tree::Y[index] as usize);

    if index == 0 || index == 18 || index == 342 || index == 360 {
        count >= 3  // corner move
    } else if x == 0 || x == 18 || y == 0 || y == 18 {
        count >= 5  // edge
    } else {
        count >= 7
    }
}

/// Play the given board until the end using the policy of the neural network
/// in a greedy manner (ignoring the pass move every time) until it is scorable
/// according to the TT-rules.
/// 
/// # Arguments
/// 
/// * `server` - the server to use during evaluation
/// * `board` - the board to score
/// * `next_color` - the color of the player whose turn it is to play
/// 
pub fn greedy_score(server: &PredictGuard, board: &Board, next_color: Color) -> (Board, String) {
    let mut board = board.clone();
    let mut sgf = String::new();
    let mut current = next_color;
    let mut pass_count = 0;
    let mut count = 0;

    while count < 722 && pass_count < 2 && !board.is_scorable() {
        let result = full_forward(&server, &board, current);
        if result.is_none() {
            break
        }

        // pick the move with the largest prior value that does not fill an
        // eye
        let (_, policy) = result.unwrap();
        let index = (0..361)
            .filter(|&i| policy[i].is_finite() && !is_eye(&board, current, i))
            .max_by_key(|&i| OrderedFloat(policy[i]));

        if let Some(index) = index {
            let (x, y) = (tree::X[index] as usize, tree::Y[index] as usize);

            sgf += &format!(";{}[{}]", current, Sabaki::to_sgf(x, y));
            pass_count = 0;
            board.place(current, x, y);
        } else {  // no valid moves remaining
            sgf += &format!(";{}[]", current);
            pass_count += 1;
        }

        // continue with the next turn
        current = current.opposite();
        count += 1;
    }

    (board, sgf)
}
