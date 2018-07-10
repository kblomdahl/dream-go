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

/// Play a game against the engine and return the result of the game.
/// This is different from `self_play` because this method does not
/// perform any search and only plays stochastically according
/// to the policy network.
/// 
/// # Arguments
/// 
/// * `server` - the server to use during evaluation
/// 
fn policy_play_one(server: &PredictGuard) -> GameResult {
    let mut temperature = (*config::TEMPERATURE + 1e-3).recip();
    let mut board = Board::new(get_random_komi());
    let mut sgf = String::new();
    let mut current = Color::Black;
    let mut pass_count = 0;
    let mut count = 0;

    while pass_count < 2 && count < 722 && !board.is_scoreable() {
        let result = forward(&server, &board, current);
        if result.is_none() {
            break
        }

        let (_, policy) = result.unwrap();

        // pick a move stochastically according to its prior value with the
        // specified temperature (to priority strongly suggested moves, and
        // avoid picking _noise_ moves).
        let index = {
            let policy_sum = policy.iter()
                .filter(|p| p.is_finite())
                .map(|p| p.powf(temperature))
                .sum::<f32>();
            let threshold = policy_sum * thread_rng().gen::<f32>();
            let mut so_far = 0.0f32;
            let mut best = None;

            for i in 0..362 {
                if policy[i].is_finite() {
                    so_far += policy[i].powf(temperature);

                    if so_far >= threshold {
                        best = Some(i);
                        break
                    }
                }
            }

            best  // if nothing, then pass
        };

        if let Some(index) = index {
            if index == 361 {  // pass
                sgf += &format!(";{}[]", current);
                pass_count += 1;
            } else {  // normal move
                let (x, y) = (tree::X[index] as usize, tree::Y[index] as usize);

                sgf += &format!(";{}[{}]", current, CGoban::to_sgf(x, y));
                pass_count = 0;
                board.place(current, x, y);
            }
        } else {  // no valid moves remaining
            sgf += &format!(";{}[]", current);
            pass_count += 1;
        }

        // continue with the next turn
        temperature = min(5.0, 1.03 * temperature);
        current = current.opposite();
        count += 1;
    }

    // if the receiver has terminated then quit
    GameResult::Ended(sgf, board)
}

/// Play games against the engine and return the results of the game over
/// the returned channel. This is different from `self_play` because this
/// method does not perform any search and only plays stochastically according
/// to the policy network.
/// 
/// # Arguments
/// 
/// * `network` - the neural network to use during evaluation
/// * `num_games` - 
/// 
pub fn policy_play(network: Network, num_games: usize) -> (Receiver<GameResult>, PredictService) {
    let server = predict::service(network);
    let (sender, receiver) = channel();

    // spawn the worker threads that generate the self-play games
    let num_workers = ::std::cmp::min(*config::NUM_GAMES, num_games);
    let remaining = Arc::new(AtomicUsize::new(num_games));

    for _ in 0..num_workers {
        let remaining = remaining.clone();
        let sender = sender.clone();
        let server = server.lock().clone_static();

        thread::spawn(move || {
            while remaining.load(Ordering::Acquire) > 0 {
                remaining.fetch_sub(1, Ordering::AcqRel);

                let result = policy_play_one(&server);

                if sender.send(result).is_err() {
                    break
                }
            }
        });
    }

    (receiver, server)
}