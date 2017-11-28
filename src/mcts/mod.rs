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
use ordered_float::OrderedFloat;
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

lazy_static! {
    /// Mapping from policy index to the `x` coordinate it represents.
    static ref _X: Box<[u8]> = (0..361).map(|i| (i % 19) as u8).collect::<Vec<u8>>().into_boxed_slice();

    /// Mapping from policy index to the `y` coordinate it represents.
    static ref _Y: Box<[u8]> = (0..361).map(|i| (i / 19) as u8).collect::<Vec<u8>>().into_boxed_slice();
}

pub enum GameResult {
    Resign(String, Board, Color, f32),
    Ended(String, Board)
}

/// Returns an index from the given policy that represents a valid move
/// and if `greedy` is false is randomly choosen weighted according to
/// the value of each index in the `policy`.
/// 
/// # Arguments
/// 
/// * `board` - the board position to use when validity
/// * `color` - the color to use when checking move validity
/// * `policy` - the policy to pluck the moves from
/// * `greedy` - whether to pick the move with the highest value or randomly
///
fn choose(board: &Board, color: Color, policy: &mut [f32], greedy: bool) -> Option<usize> {
    assert_eq!(policy.len(), 362);

    // reject any moves that are not valid from the distribution
    let mut total = policy[361];

    for i in 0..361 {
        if !board.is_valid(color, _X[i] as usize, _Y[i] as usize) {
            policy[i] = 0.0;
        } else {
            total += policy[i];
        }
    }

    if greedy {
        (0..362).max_by_key(|&i| OrderedFloat(policy[i]))
    } else {
        let mut r = total * thread_rng().next_f32();

        for i in 0..362 {
            r -= policy[i];
            if r <= 1e-6 {
                return Some(i);
            }
        }

        None
    }
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

    (value, policy)
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
        let (value, mut policy) = forward(workspace, &board, current);

        if value < -0.9 {  // resign the game if the evaluation looks bad
            sgf += &format!(";{}[]", current);

            return GameResult::Resign(sgf, board, current.opposite(), -value);
        } else {
            // add some random noise to the policy to increase the entropy of
            // the self play dataset and avoid just overfitting to the current
            // policy during training.
            dirichlet::add(&mut policy, 0.03);

            // choose a random move from the policy for the first 10 turns, after
            // that play deterministically (discounting the dirichlet noise) to
            // avoid making large blunders during life or death situations.
            let policy_m = choose(&board, current, &mut policy, count >= 10);

            if policy_m.is_none() || policy_m == Some(361) {  // passing move
                sgf += &format!(";{}[]C[{0} {}]", current, value);
                pass_count += 1;

                if pass_count >= 2 {
                    return GameResult::Ended(sgf, board)
                }
            } else if let Some(policy_m) = policy_m {
                let y = _Y[policy_m] as usize;
                let x = _X[policy_m] as usize;

                sgf += &format!(";{}[{}{}]C[{0} {}]", current, SGF_LETTERS[x], SGF_LETTERS[y], value);
                pass_count = 0;

                board.place(current, x, y);
            }
        }

        current = current.opposite();
        count += 1;
    }

    GameResult::Ended(sgf, board)
}
