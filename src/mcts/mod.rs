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

pub mod asm;
mod dirichlet;
mod global_cache;
mod greedy_score;
mod policy_play;
pub mod predict;
mod self_play;
pub mod tree;
pub mod time_control;

/* -------- Exports -------- */

pub use self::greedy_score::*;
pub use self::self_play::*;
pub use self::policy_play::*;

/* -------- Code -------- */

use rand::{thread_rng, Rng};
use std::cell::UnsafeCell;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use time;

use go::util::features::{CHW_VECT_C, Features};
use go::util::score::{Score};
use go::util::sgf::*;
use go::util::symmetry;
use go::{Board, Color};
use mcts::time_control::{TimeStrategy, RolloutLimit};
use mcts::predict::{PredictService, PredictGuard, PredictRequest};
use nn::{Network, Profiler};
use util::{b85, config, min};
use mcts::asm::sum_finite_f32;
use mcts::asm::normalize_finite_f32;

pub enum GameResult {
    Resign(String, Board, Color, f32),
    Ended(String, Board)
}

impl fmt::Display for GameResult {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let now = time::now_utc();
        let iso8601 = time::strftime("%Y-%m-%dT%H:%M:%S%z", &now).unwrap();

        match *self {
            GameResult::Resign(ref sgf, ref board, winner, _) => {
                write!(fmt, "(;GM[1]FF[4]DT[{}]SZ[19]RU[Chinese]KM[{:.1}]RE[{}+Resign]{})", iso8601, board.komi(), winner, sgf)
            },
            GameResult::Ended(ref sgf, ref board) => {
                let (black, white) = board.get_score();
                let black = black as f32;
                let white = white as f32 + board.komi();
                let winner = {
                    if black > white {
                        format!("B+{:.1}", black - white)
                    } else if white > black {
                        format!("W+{:.1}", white - black)
                    } else {
                        "0".to_string()
                    }
                };

                write!(fmt, "(;GM[1]FF[4]DT[{}]SZ[19]RU[Chinese]KM[{:.1}]RE[{}]{})", iso8601, board.komi(), winner, sgf)
            }
        }
    }
}

/// Return the value and policy for the given board position, as the interpolation
/// of their value for every symmetry.
///
/// # Arguments
///
/// * `server` - the server to use for predictions
/// * `board` - the board position to evaluate
/// * `color` - the color to evaluate for
///
fn full_forward(server: &PredictGuard, board: &Board, color: Color) -> Option<(f32, Vec<f32>)> {
    let (initial_policy, indices) = create_initial_policy(board, color);
    let mut policy = initial_policy.clone();
    let mut value = 0.0f32;

    // find out which symmetries has already been calculated, and which ones has not
    let mut new_requests = vec! [];
    let mut new_symmetries = vec! [];

    for &t in &symmetry::ALL {
        if let Some((other_value, other_policy)) = global_cache::get_or_insert(board, color, t, || { None }) {
            for i in 0..362 { policy[i] += other_policy[i]; }
            value += other_value;
        } else {
            new_requests.push(PredictRequest::Ask(board.get_features::<CHW_VECT_C>(color, t)));
            new_symmetries.push(t);
        }
    }

    // calculate any symmetries that were missing, and then accumulate them
    if let Some(new_responses) = server.send_all(new_requests) {
        for (resp, t) in new_responses.into_iter().zip(new_symmetries.into_iter()) {
            if let Some((other_value, other_policy)) = resp {
                let (other_value, other_policy) = global_cache::get_or_insert(board, color, t, || {
                    let mut identity_policy = initial_policy.clone();
                    add_valid_candidates(&mut identity_policy, other_policy, &indices, t);
                    normalize_policy(&mut identity_policy);

                    Some((0.5 + 0.5 * other_value, identity_policy))
                }).unwrap();

                for i in 0..362 { policy[i] += other_policy[i]; }
                value += other_value;
            } else {
                return None;
            }
        }

        normalize_policy(&mut policy);

        Some((value * 0.125, policy))
    } else {
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
fn forward(server: &PredictGuard, board: &Board, color: Color) -> Option<(f32, Vec<f32>)> {
    let t = *thread_rng().choose(&symmetry::ALL).unwrap();

    global_cache::get_or_insert(board, color, t, || {
        // run a forward pass through the network using this transformation
        // and when we are done undo it using the opposite.
        let response = server.send(PredictRequest::Ask(board.get_features::<CHW_VECT_C>(color, t)));
        let (value, original_policy) = if let Some(x) = response {
            x.unwrap()
        } else {
            return None;
        };

        // fix-up the potentially broken policy
        let (mut policy, indices) = create_initial_policy(board, color);
        add_valid_candidates(&mut policy, original_policy, &indices, t);
        normalize_policy(&mut policy);

        Some((0.5 + 0.5 * value, policy))
    })
}

/// Returns a initial accumulator policy where all illegal moves has been set
/// to _-Inf_, as well as an symmetry elimination mapping for its indices.
///
/// # Arguments
///
/// * `board` -
/// * `color` -
///
fn create_initial_policy(
    board: &Board,
    color: Color
) -> (Vec<f32>, Vec<usize>)
{
    // mark all illegal moves as -Inf, which effectively ensures they are never selected by
    // the tree search.
    let mut workspace = [0; 368];
    let mut policy = vec! [::std::f32::NEG_INFINITY; 368];

    for i in 0..362 {
        if i == 361 || board.is_valid_mut(color, tree::X[i] as usize, tree::Y[i] as usize, &mut workspace) {
            policy[i] = 0.0;
        }
    }

    // remove any symmetric moves that does not contribute to the search.
    //
    // we do this by finding all symmeties which provides symmetric board positions,
    // then for each candidate move we find the minimum index provided by some
    // symmetry.
    let symmetries = symmetry::ALL.iter()
        .filter(|&t| symmetry::is_symmetric(board, *t))
        .collect::<Vec<_>>();
    let mut indices = vec! [0; 362];
    indices[361] = 361;

    for i in 0..361 {
        if let Some(target) = symmetries.iter().map(|t| t.apply(i)).min() {
            indices[i] = target;

            if i != target {
                policy[i] = ::std::f32::NEG_INFINITY;
            }
        } else {
            unreachable!();
        }
    }

    (policy, indices)
}

/// Copy all valid candidates moves from `src` to `dst` applying the given symmetry and
/// the symmetry elimination map.
///
/// # Arguments
///
/// * `dst` -
/// * `src` -
/// * `indices` - the symmetry elimination map
/// * `t` - the symmetry
///
fn add_valid_candidates(
    dst: &mut Vec<f32>,
    src: Vec<f32>,
    indices: &[usize],
    t: symmetry::Transform
) {
    // always copy the _passing_ move since it is never an illegal move.
    dst[361] += src[361];

    // de-transform each index in the source policy, to the identity board position
    // before adding it to the destination.
    for i in 0..361 {
        let j = indices[t.inverse().apply(i)];

        dst[j] += src[i];
    }
}

/// Normalize the given vector so that its elements sums to `1.0`.
///
/// # Arguments
///
/// * `policy` - the vector to normalize in-place
///
fn normalize_policy(policy: &mut Vec<f32>) {
    // re-normalize the policy since we have modified its values
    let policy_sum: f32 = sum_finite_f32(&policy);

    if policy_sum < 1e-6 {  // do not divide by zero
        dirichlet::add_ex(&mut policy[0..362], 0.03, 1.0);
    } else {
        normalize_finite_f32(policy, policy_sum);
    }

    // check for NaN
    for i in 0..362 {
        assert!(!policy[i].is_nan(), "found NaN at index {}, total sum = {}", i, policy_sum);
    }
}

/// The shared variables between the master and each worker thread in the `predict` function.
#[derive(Clone)]
struct ThreadContext<T: TimeStrategy + Clone + Send> {
    /// The root of the monte carlo tree.
    root: Arc<UnsafeCell<tree::Node>>,

    /// The initial board position at the root the tree.
    starting_point: Board,

    /// Time control element
    time_strategy: T,
}

unsafe impl<T: TimeStrategy + Clone + Send> Send for ThreadContext<T> { }


/// Worker that probes into the given monte carlo search tree until the context
/// is exhausted.
///
/// # Arguments
///
/// * `context` -
/// * `server` -
///
fn predict_worker<T>(context: ThreadContext<T>, server: PredictGuard)
    where T: TimeStrategy + Clone + Send + 'static
{
    let root = unsafe { &mut *context.root.get() };

    while !time_control::is_done(root, &context.time_strategy) {
        loop {
            let mut board = context.starting_point.clone();
            let trace = unsafe { tree::probe(root, &mut board) };

            if let Some(trace) = trace {
                let &(_, color, _) = trace.last().unwrap();
                let next_color = color.opposite();
                let result = forward(&server, &board, next_color);

                if let Some((value, policy)) = result {
                    unsafe {
                        tree::insert(&trace, next_color, value, policy);
                        break
                    }
                } else {
                    return  // unrecognized error
                }
            } else {
                server.send(PredictRequest::Wait);
            }
        }
    }
}

/// Predicts the _best_ next move according to the given neural network when applied
/// to a monte carlo tree search.
///
/// # Arguments
///
/// * `server` - the server to use during evaluation
/// * `num_workers` -
/// * `starting_tree` -
/// * `starting_point` -
/// * `starting_color` -
///
fn predict_aux<T>(
    server: &PredictGuard,
    num_workers: usize,
    time_strategy: T,
    starting_tree: Option<tree::Node>,
    starting_point: &Board,
    starting_color: Color
) -> (f32, usize, tree::Node)
    where T: TimeStrategy + Clone + Send + 'static
{
    let (starting_value, mut starting_policy) =
        full_forward(&server, starting_point, starting_color)
            .unwrap_or_else(|| {
                let mut policy = vec! [0.0; 362];
                policy[361] = 1.0;

                (0.5, policy)
            });

    // add some dirichlet noise to the root node of the search tree in order to increase
    // the entropy of the search and avoid overfitting to the prior value
    dirichlet::add(&mut starting_policy[..362], 0.03);

    // if we have a starting tree given, then re-use that tree (after some sanity
    // checks), otherwise we need to query the neural network about what the
    // prior value should be at the root node.
    let starting_tree = if let Some(mut starting_tree) = starting_tree {
        assert_eq!(starting_tree.color, starting_color);

        // replace the prior value of the tree, since it was either:
        //
        // - calculated using only one symmetry.
        // - a pre-expanded pass move, which does not get a prior computed.
        //
        starting_tree.prior[0..362].clone_from_slice(&starting_policy[..362]);
        starting_tree
    } else {
        tree::Node::new(starting_color, starting_value, starting_policy)
    };

    // start-up all of the worker threads, and then start listening for requests on the
    // channel we gave each thread.
    let context: ThreadContext<T> = ThreadContext {
        root: Arc::new(UnsafeCell::new(starting_tree)),
        starting_point: starting_point.clone(),

        time_strategy: time_strategy.clone()
    };

    if num_workers <= 1 {
        let context = context.clone();
        let server = server.clone_static();

        predict_worker::<T>(context, server);
    } else {
        let handles = (0..num_workers).map(|_| {
            let context = context.clone();
            let server = server.clone_static();

            thread::spawn(move || predict_worker::<T>(context, server))
        }).collect::<Vec<JoinHandle<()>>>();

        // wait for all threads to terminate to avoid any zombie processes
        for handle in handles.into_iter() { handle.join().unwrap(); }
    }

    assert_eq!(Arc::strong_count(&context.root), 1);

    // choose the best move according to the search tree
    let root = UnsafeCell::into_inner(Arc::try_unwrap(context.root).ok().expect(""));
    let (value, index) = root.best(if starting_point.count() < 8 {
        *config::TEMPERATURE
    } else {
        0.0
    });

    #[cfg(feature = "trace-mcts")]
    eprintln!("{}", tree::to_sgf::<CGoban>(&root, starting_point, true));

    (value, index, root)
}

/// Predicts the _best_ next move according to the given neural network when applied
/// to a monte carlo tree search.
///
/// # Arguments
///
/// * `server` - the server to use during evaluation
/// * `num_workers` -
/// * `starting_tree` -
/// * `starting_point` -
/// * `starting_color` -
///
pub fn predict<T>(
    server: &PredictGuard,
    num_workers: Option<usize>,
    time_control: T,
    starting_tree: Option<tree::Node>,
    starting_point: &Board,
    starting_color: Color
) -> (f32, usize, tree::Node)
    where T: TimeStrategy + Clone + Send + 'static
{
    let num_workers = num_workers.unwrap_or(*config::NUM_THREADS);

    Profiler::with(move || {
        predict_aux::<T>(server, num_workers, time_control, starting_tree, starting_point, starting_color)
    })
}

/// Returns a weighted random komi between `-7.5` to `7.5`, with the most common
/// ones being `7.5`, `6.5`, and `0.5`.
///
/// - 40% chance of `7.5`
/// - 40% chance of `6.5`
/// - 10% chance of `0.5`
/// - 10% chance of a random komi between `-7.5` and `7.5`.
///
fn get_random_komi() -> f32 {
    let value = thread_rng().gen::<f32>();

    if value < 0.4 {
        7.5
    } else if value < 0.8 {
        6.5
    } else if value < 0.9 {
        0.5
    } else {
        let value = thread_rng().gen_range::<i32>(-8, 8);

        value as f32 + 0.5
    }
}

#[cfg(test)]
mod tests {
    use mcts;

    #[test]
    fn get_random_komi() {
        // i do not like the use of randomness in tests, but I do not see much
        // choice here
        for _ in 0..10000 {
            let komi = mcts::get_random_komi();

            assert!(komi >= -7.5 && komi <= 7.5, "komi is {}", komi);
        }
    }
}
