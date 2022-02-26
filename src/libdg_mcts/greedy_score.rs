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

use dg_go::{Board, Color, Point};
use dg_sgf::{CGoban, ToSgf};
use super::predictor::Predictor;
use super::{full_forward, ScoringSearch, SearchOptions};


/// Play the given board until the end using the policy of the neural network
/// in a greedy manner (ignoring the pass move every time) until it is scorable
/// according to the TT-rules.
///
/// # Arguments
///
/// * `server` - the server to use during evaluation
/// * `board` - the board to score
/// * `to_move` - the color of the player whose turn it is to play
///
pub fn greedy_score(predictor: &dyn Predictor, board: &Board, mut to_move: Color) -> (Board, String) {
    let options: Box<dyn SearchOptions + Sync> = Box::new(ScoringSearch::default());
    let mut board = board.clone();
    let mut sgf = String::new();
    let mut pass_count = 0;
    let mut count = 0;

    while count < 722 && pass_count < 2 {
        let policy = if let Some(response) = full_forward(predictor, &options, &board, to_move) {
            response.1
        } else {
            return (board, sgf)
        };

        // pick the move with the largest prior value that does not fill an
        // eye
        let index = (0..361)
            .filter(|&i| policy[i].is_finite())
            .max_by_key(|&i| OrderedFloat(policy[i]));

        if let Some(index) = index {
            let point = Point::from_packed_parts(index);

            sgf += &format!(";{}[{}]", to_move, point.to_sgf::<CGoban>());
            pass_count = 0;
            board.place(to_move, point);
        } else {  // no valid moves remaining
            sgf += &format!(";{}[]", to_move);
            pass_count += 1;
        }

        // continue with the next turn
        to_move = to_move.opposite();
        count += 1;
    }

    (board, sgf)
}
