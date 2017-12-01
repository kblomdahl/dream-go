// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

mod dirichlet;
mod tree;

use rand::{thread_rng, Rng};

use ::go::{symmetry, Board, Color};
use ::nn::{self, Workspace};

/// Mapping from 1D coordinate to letter used to represent that coordinate in
/// the SGF file format.
const SGF_LETTERS: [char; 26] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
    'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
    'w', 'x', 'y', 'z'
];

pub enum GameResult {
    Resign(String, Board, Color, f32),
    Ended(String, Board)
}

/// Performs a forward pass through the neural network for the given board
/// position using a random symmetry to increase entropy.
/// 
/// # Arguments
/// 
/// * `workspace` - the workspace to use during the forward pass
/// * `board` - the board position
/// * `color` - the current player
/// 
fn forward(workspace: &mut Workspace, board: &Board, color: Color) -> (f32, Box<[f32]>) {
    lazy_static! {
        static ref SYMM: Vec<symmetry::Transform> = vec! [
            symmetry::Transform::Identity,
            symmetry::Transform::FlipLR,
            symmetry::Transform::FlipUD,
            symmetry::Transform::Transpose,
            symmetry::Transform::TransposeAnti,
            symmetry::Transform::Rot90,
            symmetry::Transform::Rot180,
            symmetry::Transform::Rot270,
        ];
    }

    // pick a random transformation to apply to the features. This is done
    // to increase the entropy of the game slightly and to ensure the engine
    // learns the game is symmetric (which should help generalize)
    let t = *thread_rng().choose(&SYMM).unwrap();
    let mut features = board.get_features(color);

    symmetry::apply(&mut features, t);

    // run a forward pass through the network using this transformation
    // and when we are done undo it using the opposite.
    let (value, mut policy) = nn::forward(workspace, &features);

    symmetry::apply(&mut policy, t.inverse());

    // replace any invalid moves in the suggested policy with -Inf, while keeping
    // the pass move (361) untouched so that there is always at least one valid
    // move.
    for i in 0..361 {
        let (x, y) = (tree::X[i] as usize, tree::Y[i] as usize);

        if !board.is_valid(color, x, y) {
            policy[i] = ::std::f32::NEG_INFINITY;
        }
    }

    (value, policy)
}

pub fn predict(workspace: &mut Workspace, starting_point: &Board, starting_color: Color) -> (f32, usize, usize) {
    // add some dirichlet noise to the root node of the search tree in order to increase
    // the entropy of the search and avoid overfitting to the prior value
    let (_, mut policy) = forward(workspace, starting_point, starting_color);
    dirichlet::add(&mut policy, 0.03);

    // perform exactly 2,000 probes into the search tree
    let mut root = tree::Node::new(starting_color, policy);

    for _ in 0..2000 {
        let mut board = starting_point.clone();
        let trace = unsafe { tree::probe(&mut root, &mut board) };

        if let Some(&(_, color, _)) = trace.last() {
            let next_color = color.opposite();
            let (value, policy) = forward(workspace, &board, next_color);

            unsafe {
                tree::insert(&trace, next_color, value, policy);
            }
        }
    }

    let (value, index) = root.best();
    let (_, prior_index) = root.prior();

    (value, index, prior_index)
}

/// Play a game against the engine and return the result of the game.
/// 
/// # Arguments
/// 
/// * `workspace` - the neural network workspace to use during evaluation
/// 
pub fn self_play(workspace: &mut Workspace) -> GameResult {
    let mut board = Board::new();
    let mut sgf = String::new();
    let mut current = Color::Black;
    let mut pass_count = 0;
    let mut count = 0;

    // limit the maximum number of moves to `2 * 19 * 19` to avoid the
    // engine playing pointless capture sequences at the end of the game
    // that does not change the final result.
    while count < 722 {
        let (value, index, prior_index) = predict(workspace, &board, current);

        debug_assert!(-1.0 <= value && value <= 1.0);
        debug_assert!(index < 362);

        if value < -0.9 {  // resign the game if the evaluation looks bad
            sgf += &format!(";{}[]", current);

            return GameResult::Resign(sgf, board, current.opposite(), -value);
        } else {
            if index == 361 {  // passing move
                sgf += &format!(";{}[]C[{0} {}]", current, value);
                pass_count += 1;

                if pass_count >= 2 {
                    return GameResult::Ended(sgf, board)
                }
            } else {
                let (x, y) = (tree::X[index] as usize, tree::Y[index] as usize);
                let (px, py) = (tree::X[prior_index] as usize, tree::Y[prior_index] as usize);

                sgf += &format!(";{}[{}{}]TR[{}{}]C[{0} {}]",
                    current,
                    SGF_LETTERS[x], SGF_LETTERS[y],
                    SGF_LETTERS[px], SGF_LETTERS[py],
                    value
                );
                pass_count = 0;

                board.place(current, x, y);
            }
        }

        current = current.opposite();
        count += 1;
    }

    GameResult::Ended(sgf, board)
}
