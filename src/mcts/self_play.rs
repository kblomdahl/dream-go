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

use go::{Board, Color};
use util::config;
use mcts::*;

use std::mem;

/// Play a game against the engine and return the result of the game.
///
/// # Arguments
///
/// * `server` - the server to use during evaluation
/// * `num_parallel` - the number of games that are being played in parallel
///
fn self_play_one(server: &PredictGuard, num_parallel: &Arc<AtomicUsize>) -> GameResult
{
    let mut board = Board::new(get_random_komi());
    let mut sgf = String::new();
    let mut current = Color::Black;
    let mut pass_count = 0;
    let mut count = 0;

    // limit the maximum number of moves to `2 * 19 * 19` to avoid the
    // engine playing pointless capture sequences at the end of the game
    // that does not change the final result.
    let allow_resign = thread_rng().gen::<f32>() < 0.95;
    let mut root_current = None;
    let mut root_other = None;

    while count < 722 {
        let num_workers = *config::NUM_THREADS / num_parallel.load(Ordering::Acquire);
        let (value, index, tree) = predict_aux::<_>(
            &server,
            num_workers,
            RolloutLimit::new(*config::NUM_ROLLOUT),
            root_current,
            &board,
            current
        );

        debug_assert!(0.0 <= value && value <= 1.0);
        debug_assert!(index < 362);

        let policy = tree.softmax();
        let (_, prior_index) = tree.prior();
        let value_sgf = if current == Color::Black { 2.0 * value - 1.0 } else { -2.0 * value + 1.0 };

        if allow_resign && value < 0.05 {  // resign the game if the evaluation looks bad
            return GameResult::Resign(sgf, board, current.opposite(), -value);
        } else if index == 361 {  // passing move
            sgf += &format!(";{}[]P[{}]V[{}]", current, b85::encode(&policy), value_sgf);
            pass_count += 1;

            if pass_count >= 2 {
                return GameResult::Ended(sgf, board)
            }
        } else {
            let (x, y) = (tree::X[index] as usize, tree::Y[index] as usize);

            sgf += &format!(";{}[{}]P[{}]V[{}]",
                current,
                CGoban::to_sgf(x, y),
                b85::encode(&policy),
                value_sgf
            );
            if prior_index != 361 {
                sgf += &format!("TR[{}]",
                    CGoban::to_sgf(
                        tree::X[prior_index] as usize,
                        tree::Y[prior_index] as usize
                    )
                );
            };

            pass_count = 0;
            board.place(current, x, y);
        }

        // update the search trees
        root_current = tree::Node::forward(tree, index);
        root_other = if let Some(other) = root_other {
            tree::Node::forward(other, index)
        } else {
            None
        };

        // swap whose turn it is to place a stone
        mem::swap(&mut root_current, &mut root_other);
        current = current.opposite();
        count += 1;
    }

    GameResult::Ended(sgf, board)
}

/// Play games against the engine and return the result of the games
/// over the channel.
///
/// # Arguments
///
/// * `network` - the neural network to use during evaluation
/// * `num_games` - the number of games to generate
///
pub fn self_play(network: Network, num_games: usize) -> (Receiver<GameResult>, PredictService) {
    let server = predict::service(network);
    let (sender, receiver) = channel();

    // spawn the worker threads that generate the self-play games
    let num_parallel = ::std::cmp::min(num_games, *config::NUM_GAMES);
    let num_workers = Arc::new(AtomicUsize::new(num_parallel));
    let processed = Arc::new(AtomicUsize::new(0));

    for _ in 0..num_parallel {
        let num_workers = num_workers.clone();
        let processed = processed.clone();
        let sender = sender.clone();
        let server = server.lock().clone_static();

        thread::spawn(move || {
            while processed.fetch_add(1, Ordering::SeqCst) < num_games {
                let result = self_play_one(&server, &num_workers);

                if sender.send(result).is_err() {
                    break
                }
            }

            num_workers.fetch_sub(1, Ordering::Release);
        });
    }

    (receiver, server)
}
