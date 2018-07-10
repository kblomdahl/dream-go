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

use ordered_float::OrderedFloat;

use go::{Board, Color};
use mcts::*;

/// Play the given board until the end using the policy of the neural network
/// in a greedy manner (ignoring the pass move every time) until it is scoreable
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

    while count < 722 && pass_count < 2 && !board.is_scoreable() {
        let result = forward(&server, &board, current);
        if result.is_none() {
            break
        }

        let (_, policy) = result.unwrap();

        // pick a move stochastically according to its prior value with the
        // specified temperature (to priority strongly suggested moves, and
        // avoid picking _noise_ moves).
        let index = (0..361)
            .filter(|&i| policy[i].is_finite())
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