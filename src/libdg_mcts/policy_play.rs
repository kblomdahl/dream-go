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
use rand::{thread_rng, Rng};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::thread;

use dg_go::utils::sgf::{CGoban, SgfCoordinate};
use dg_go::{Board, Color, Point};
use dg_utils::{b85, config, min};
use super::asm::sum_finite_f32;
use super::predict::Predictor;
use super::time_control::RolloutLimit;
use super::{dirichlet, predict_service};
use super::{GameResult, full_forward, get_random_komi, predict_aux};
use dg_nn::Network;
use options::StandardSearch;

/// Returns the skewness of the given policy. A large return value says that
/// the given input `policy` is less certain, and therefore more interesting
/// to search on.
///
/// # Arguments
///
/// * `policy` -
///
fn skewness(policy: &[f32]) -> f32 {
    let mean = sum_finite_f32(&policy) / (policy.len() as f32);

    // calculate the second and third moments
    let k_3 = policy.iter()
        .filter(|&p| p.is_finite())
        .map(|&p| (p - mean).powi(3))
        .sum::<f32>() / (policy.len() as f32);
    let k_2 = policy.iter()
        .filter(|&p| p.is_finite())
        .map(|&p| (p - mean).powi(2))
        .sum::<f32>() / (policy.len() as f32 - 1.0);

    // calculate the skew, and then inverse it so that a low skew becomes _good_
    let b = k_3 / k_2.powf(3.0 / 2.0);

    1.0 / b.powi(2)
}

/// pick the move from the policy by taking the top 80% of the policy, and then
/// choosing a move from the remains (weighted by the moves policy value)
///
/// # Arguments
///
/// * `policy` -
/// * `temperature` -
///
fn policy_choose(policy: &[f32], temperature: f32) -> Option<usize> {
    let mut candidates = (0..362).collect::<Vec<usize>>();
    let mut subtotals = [0.0f32; 362];

    candidates.sort_unstable_by_key(|&i| OrderedFloat(-policy[i]));

    // calculate the subtotals for the sorted candidates so that we can
    // efficiently determine the cutoff point, using binary search.
    for i in 0..362 {
        let j = candidates[i];
        let value = policy[j].powf(temperature);

        if value.is_finite() {
            subtotals[i] = if i > 0 { subtotals[i-1] + value } else { value };
        } else {
            subtotals[i] = if i > 0 { subtotals[i-1] } else { 0.0 };
        }
    }

    // if there are no valid moves remaining then pass, otherwise pick
    // a random move using binary search over the `subtotals`.
    if subtotals[361] > 0.0 {
        let threshold = 0.8 * thread_rng().gen::<f32>() * subtotals[361];
        let mut index = match subtotals.binary_search_by_key(&OrderedFloat(threshold), |&s| OrderedFloat(s)) {
            Ok(i) => i,
            Err(i) => i
        };

        // if the binary search found one of the invalid moves then step
        // backward, and then forward again iff we landed on a move with
        // no preceeding legal move.
        while index > 0 && subtotals[index - 1] == subtotals[index] {
            index -= 1;
        }

        while subtotals[index] == 0.0 {
            index += 1;
        }

        Some(candidates[index])
    } else {
        None
    }
}

fn policy_ex_it<P: Predictor + 'static>(server: &P, board: &Board, to_move: Color) -> Option<(String, f32)> {
    let (value, _index, tree) = predict_aux::<_, _, StandardSearch>(
        server,
        1,
        RolloutLimit::new((*config::NUM_ROLLOUT).into()),
        None,
        board,
        to_move
    )?;

    let policy_sgf = b85::encode(&tree.softmax());
    let value_sgf = if to_move == Color::Black {
        2.0 * value - 1.0
    } else {
        -2.0 * value + 1.0
    };

    Some((policy_sgf, value_sgf))
}

/// Perform a _search_ of the given board state, using the number of rollouts
/// specified in `NUM_POLICY_ROLLOUT`.
///
/// # Arguments
///
/// * `server` -
/// * `board` -
/// * `to_move` -
///
fn policy_forward<P: Predictor + 'static>(
    server: &P,
    board: &Board,
    to_move: Color
) -> Option<(f32, Vec<f32>)>
{
    let num_policy_rollout = *config::NUM_POLICY_ROLLOUT;

    if num_policy_rollout <= 1 {
        let (value, mut policy) = full_forward::<_, StandardSearch>(server, board, to_move)?;
        dirichlet::add(&mut policy[0..362], 0.03);

        Some((value, policy))
    } else {
        let (value, _index, tree) = predict_aux::<_, _, StandardSearch>(
            server,
            1,
            RolloutLimit::new(num_policy_rollout),
            None,
            &board,
            to_move
        )?;

        Some((value, tree.softmax()))
    }
}

/// Play a game against the engine and return the result of the game.
/// This is different from `self_play` because this method does not
/// perform any search and only plays stochastically according
/// to the policy network.
///
/// # Arguments
///
/// * `server` - the server to use during evaluation
/// * `ex_it` - whether to emit one full policy
///
fn policy_play_one<P: Predictor + 'static>(server: &P, ex_it: bool) -> Option<GameResult> {
    let mut temperature = (*config::TEMPERATURE + 1e-3).recip();
    let mut sgf = vec! [];

    // loop until we run or of legal moves, the board is fully scorable, or
    // we have played 722 moves in total.
    let mut board = Board::new(get_random_komi());
    let mut color = Color::Black;
    let mut pass_count = 0;
    let mut total_skew = 0.0;

    while pass_count < 2 && board.count() < 722 {
        let (index, skew) = {
            let (_value, policy) = policy_forward(server, &board, color)?;

            match policy_choose(&policy, temperature) {
                Some(index) => (index, skewness(&policy)),
                None => (361, 0.0)
            }
        };

        if index == 361 {
            sgf.push((board.clone(), color, skew, format!(";{}[]", color)));
            pass_count += 1;
        } else {
            let point = Point::from_packed_parts(index);

            sgf.push((board.clone(), color, skew, format!(";{}[{}]", color, CGoban::to_sgf(point))));
            board.place(color, point);
            pass_count = 0;
        }

        total_skew += skew;
        temperature = min(5.0, 1.03 * temperature);
        color = color.opposite();
    }

    // if we are running with --ex-it then we need compute policies for some
    // moves.
    if ex_it {
        // reject the top 50% most skewed prior values, since they will not produce useful
        // search trees anyway.
        let mut indices = (0..sgf.len()).collect::<Vec<_>>();
        let num_samples = match *config::NUM_SAMPLES {
            config::SamplingStrategy::Percent(pct) => (pct * indices.len() as f32) as usize,
            config::SamplingStrategy::Fixed(num) => ::std::cmp::min(num, indices.len())
        };

        for _j in 0..num_samples {
            let cutoff = thread_rng().gen::<f32>() * total_skew;
            let mut so_far = 0.0;
            let mut i;
            let mut j = 0;

            loop {
                i = indices[j];
                so_far += sgf[i].2;

                if so_far >= cutoff {
                    break
                }

                j += 1;
            }

            // for each `i`, compute the _true_ policy using MCTS
            let (policy_sgf, value_sgf) = policy_ex_it(server, &sgf[i].0, sgf[i].1)?;

            sgf[i].3 = format!("{}P[{}]V[{:.4}]", sgf[i].3, policy_sgf, value_sgf);

            // remove the sample from the available samples so that we do not compute it twice
            total_skew -= sgf[i].2;
            indices.swap_remove(j);
        }
    }

    Some(GameResult::Ended(
        sgf.into_iter().fold(String::new(), |acc, (_board, _color, _skew, sgf)| acc + &sgf),
        board
    ))
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
/// * `ex_it` - whether to emit one full policy per game
///
pub fn policy_play(network: Network, num_games: usize, ex_it: bool) -> (Receiver<GameResult>, predict_service::PredictService) {
    let server = predict_service::service(network);
    let (sender, receiver) = channel();

    // spawn the worker threads that generate the self-play games
    let num_workers = ::std::cmp::min(*config::NUM_GAMES, num_games);
    let remaining = Arc::new(AtomicUsize::new(num_games));

    for _ in 0..num_workers {
        let remaining = remaining.clone();
        let sender = sender.clone();
        let server = server.lock().clone_to_static();

        thread::spawn(move || {
            while remaining.load(Ordering::Acquire) > 0 {
                remaining.fetch_sub(1, Ordering::AcqRel);

                if let Some(result) = policy_play_one(&server, ex_it) {
                    if sender.send(result).is_err() {
                        break
                    }
                }
            }
        });
    }

    (receiver, server)
}
