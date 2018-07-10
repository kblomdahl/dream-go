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

mod argmax;
mod dirichlet;
mod global_cache;
mod greedy_score;
mod policy_play;
pub mod predict;
mod self_play;
mod spin;
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
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use time;

use go::sgf::*;
use go::{symmetry, Board, Color, CHW_VECT_C, Features, Score};
use mcts::time_control::{TimeStrategy, RolloutLimit};
use mcts::predict::{PredictService, PredictGuard, PredictRequest};
use nn::Network;
use util::b85;
use util::config;
use util::min;

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
                        format!("0")
                    }
                };

                write!(fmt, "(;GM[1]FF[4]DT[{}]SZ[19]RU[Chinese]KM[{:.1}]RE[{}]{})", iso8601, board.komi(), winner, sgf)
            }
        }
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
fn forward(server: &PredictGuard, board: &Board, color: Color) -> Option<(f32, Box<[f32]>)> {
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

    global_cache::get_or_insert(board, color, || {
        // pick a random transformation to apply to the features. This is done
        // to increase the entropy of the game slightly and to ensure the engine
        // learns the game is symmetric (which should help generalize)
        let t = *thread_rng().choose(&SYMM).unwrap();

        // run a forward pass through the network using this transformation
        // and when we are done undo it using the opposite.
        let response = server.send(PredictRequest::Ask(board.get_features::<CHW_VECT_C>(color, t)));
        let (value, original_policy) = if let Some(x) = response {
            x.unwrap()
        } else {
            return None;
        };

        // copy the policy and replace any invalid moves in the suggested policy
        // with -Inf, while keeping the pass move (361) untouched so that there
        // is always at least one valid move.
        let mut policy = vec! [0.0f32; 362];
        policy[361] = original_policy[361];  // copy `pass` move

        for i in 0..361 {
            let j = t.inverse().apply(i);
            let (x, y) = (tree::X[j] as usize, tree::Y[j] as usize);

            if !board.is_valid(color, x, y) {
                policy[j] = ::std::f32::NEG_INFINITY;
            } else {
                policy[j] = original_policy[i];
            }
        }

        // get ride of symmetric moves, this is mostly useful for the opening.
        // Once we are past the first ~7 moves the board is usually sufficiently
        // asymmetric for this to turn into a no-op.
        //
        // we skip the first symmetry because it is the identity symmetry, which
        // is always a symmetry for any board.
        for &t in &SYMM[1..8] {
            if !symmetry::is_symmetric(board, t) {
                continue;
            }

            // figure out which are the useful vertices by eliminating the
            // symmetries from the board.
            let mut visited = [false; 368];

            for i in 0..361 {
                let j = t.apply(i);

                if i != j && !visited[i] {
                    visited[i] = true;
                    visited[j] = true;

                    let src = ::std::cmp::max(i, j);
                    let dst = ::std::cmp::min(i, j);

                    if policy[src].is_finite() {
                        assert!(policy[dst].is_finite());

                        policy[dst] += policy[src];
                        policy[src] = ::std::f32::NEG_INFINITY;
                    }
                }
            }
        }

        // renormalize the policy so that it sums to one after all the pruning that
        // we have performed.
        let mut policy_sum: f32 = policy.iter().filter(|p| p.is_finite()).sum();

        if policy_sum < 1e-6 {  // do not divide by zero
            policy_sum = 0.0;

            for i in 0..362 {
                if policy[i].is_finite() {
                    let value = thread_rng().gen();

                    policy[i] = value;
                    policy_sum += value;
                }
            }
        }

        let policy_recip = policy_sum.recip();

        for i in 0..362 {
            policy[i] *= policy_recip;
        }

        Some((0.5 * value + 0.5, policy.into_boxed_slice()))
    })
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

    /// The number of probes that still needs to be done into the tree.
    remaining: Arc<AtomicIsize>,
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
    // if we have a starting tree given, then re-use that tree (after some sanity
    // checks), otherwise we need to query the neural network about what the
    // prior value should be at the root node.
    let mut starting_tree = if let Some(mut starting_tree) = starting_tree {
        assert_eq!(starting_tree.color, starting_color);

        if starting_tree.prior.iter().sum::<f32>() < 1e-4 {
            // we are missing the prior distribution, this can happend if we
            // fast-forwarded a passing move, but the pass move had not been
            // expanded (since we still need to create the node to record
            // that it was a pass so that we do not lose count of the number
            // of consecutive passes).
            let server = server.clone();
            let (_, policy) = forward(&server, starting_point, starting_color)
                .unwrap_or_else(|| {
                    let mut policy = vec! [0.0; 362];
                    policy[361] = 1.0;

                    (0.5, policy.into_boxed_slice())
                });

            for i in 0..362 {
                starting_tree.prior[i] = policy[i];
            }
        }

        starting_tree
    } else {
        let server = server.clone();
        let (value, mut policy) = forward(&server, starting_point, starting_color)
            .unwrap_or_else(|| {
                let mut policy = vec! [0.0; 362];
                policy[361] = 1.0;

                (0.5, policy.into_boxed_slice())
            });

        tree::Node::new(starting_color, value, policy)
    };

    // add some dirichlet noise to the root node of the search tree in order to increase
    // the entropy of the search and avoid overfitting to the prior value
    dirichlet::add(&mut starting_tree.prior, 0.03);

    // start-up all of the worker threads, and then start listening for requests on the
    // channel we gave each thread.
    let remaining = if *config::NUM_ROLLOUT > starting_tree.size() {
        (*config::NUM_ROLLOUT - starting_tree.size()) as isize
    } else {
        0
    };
    let context: ThreadContext<T> = ThreadContext {
        root: Arc::new(UnsafeCell::new(starting_tree)),
        starting_point: starting_point.clone(),

        time_strategy: time_strategy.clone(),
        remaining: Arc::new(AtomicIsize::new(remaining)),
    };

    let handles = (0..num_workers).map(|_| {
        let context = context.clone();
        let server = server.clone_static();

        thread::spawn(move || predict_worker::<T>(context, server))
    }).collect::<Vec<JoinHandle<()>>>();

    // wait for all threads to terminate to avoid any zombie processes
    for handle in handles.into_iter() { handle.join().unwrap(); }

    assert_eq!(Arc::strong_count(&context.root), 1);

    // choose the best move according to the search tree
    let root = UnsafeCell::into_inner(Arc::try_unwrap(context.root).ok().expect(""));
    let (value, index) = root.best(if starting_point.count() < 8 {
        *config::TEMPERATURE
    } else {
        0.0
    });

    #[cfg(feature = "trace-mcts")]
    eprintln!("{}", tree::to_sgf::<CGoban, E>(&root, starting_point, true));

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

    predict_aux::<T>(server, num_workers, time_control, starting_tree, starting_point, starting_color)
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
