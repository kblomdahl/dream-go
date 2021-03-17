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

#![feature(core_intrinsics)]
#![feature(test)]

extern crate crossbeam_channel;
extern crate crossbeam_utils;
extern crate crossbeam_queue;
extern crate dg_cuda;
extern crate dg_go;
extern crate dg_nn;
extern crate dg_utils;
#[macro_use] extern crate lazy_static;
extern crate ordered_float;
extern crate rand;
extern crate rand_distr;
#[cfg(test)] extern crate test;
extern crate time;

/* -------- Modules -------- */

pub mod asm;
mod choose;
mod dirichlet;
mod game_result;
mod global_cache;
mod greedy_score;
pub mod options;
pub mod parallel;
pub mod predict_service;
pub mod predict;
mod reanalyze;
mod self_play;
pub mod tree;
pub mod time_control;

/* -------- Exports -------- */

pub use self::game_result::*;
pub use self::greedy_score::*;
pub use self::self_play::*;
pub use self::reanalyze::*;

/* -------- Code -------- */

use rand::prelude::SliceRandom;
use rand::{thread_rng, Rng};
use std::cell::UnsafeCell;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use dg_go::utils::features::{HWC, Features};
use dg_go::utils::symmetry;
use dg_go::{Board, Color, Point};
use self::options::{SearchOptions, ScoringSearch};
use self::time_control::TimeStrategy;
use self::tree::ProbeResult;
use self::predict::Predictor;
use dg_utils::config;
use dg_utils::types::f16;
use self::asm::sum_finite_f32;
use self::asm::normalize_finite_f32;
use self::parallel::global_rwlock;

/// Return the value and policy for the given board position, as the interpolation
/// of their value for every symmetry.
///
/// # Arguments
///
/// * `server` - the server to use for predictions
/// * `options` - 
/// * `board` - the board position to evaluate
/// * `to_move` - the color to evaluate for
///
fn full_forward<P: Predictor>(server: &P, options: &dyn SearchOptions, board: &Board, to_move: Color) -> Option<(f32, Vec<f32>)> {
    let (initial_policy, indices) = create_initial_policy(options, board, to_move);
    let mut policy = initial_policy.clone();
    let mut value = 0.0f32;

    // find out which symmetries has already been calculated, and which ones has not
    let mut new_requests = Vec::with_capacity(8);
    let mut new_symmetries = Vec::with_capacity(8);

    for &t in &symmetry::ALL {
        if let Some((other_value, other_policy)) = global_cache::get_or_insert(board, to_move, t, || { None }) {
            for i in 0..362 { policy[i] += other_policy[i]; }
            value += other_value;
        } else {
            new_requests.push(board.get_features::<HWC, f16>(to_move, t));
            new_symmetries.push(t);
        }
    }

    // calculate any symmetries that were missing, add them to the cache, and then take the
    // average of them
    let new_responses = server.predict_all(new_requests.into_iter());

    for (new_response, t) in new_responses.into_iter().zip(new_symmetries.into_iter()) {
        let (other_value, other_policy) = new_response?;
        let (other_value, other_policy) = global_cache::get_or_insert(board, to_move, t, || {
            let mut identity_policy = initial_policy.clone();
            add_valid_candidates(&mut identity_policy, other_policy, &indices, t);
            normalize_policy(&mut identity_policy);

            Some((0.5 + 0.5 * other_value, identity_policy))
        }).unwrap();

        for i in 0..362 { policy[i] += other_policy[i]; }
        value += other_value;
    }

    normalize_policy(&mut policy);

    Some((value * 0.125, policy))
}

/// Performs a forward pass through the neural network for the given board
/// position using a random symmetry to increase entropy.
///
/// # Arguments
///
/// * `server` - the workspace to use during the forward pass
/// * `options` - 
/// * `board` - the board position
/// * `to_move` - the current player
///
fn forward<P: Predictor>(server: &P, options: &dyn SearchOptions, board: &Board, to_move: Color) -> Option<(f32, Vec<f32>)> {
    let t = *symmetry::ALL.choose(&mut thread_rng()).unwrap();

    global_cache::get_or_insert(board, to_move, t, || {
        // run a forward pass through the network using this transformation
        // and when we are done undo it using the opposite.
        let (value, original_policy) = server.predict(
            board.get_features::<HWC, f16>(
                to_move,
                t
            )
        )?;

        // fix-up the potentially broken policy
        let (mut policy, indices) = create_initial_policy(options, board, to_move);
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
fn create_initial_policy(options: &dyn SearchOptions, board: &Board, to_move: Color) -> (Vec<f32>, Vec<usize>) {
    // mark all illegal moves as -Inf, which effectively ensures they are never selected by
    // the tree search.
    let mut policy = vec! [::std::f32::NEG_INFINITY; 368];
    let policy_checker = options.policy_checker(board, to_move);

    for point in Point::all() {
        if policy_checker.is_policy_candidate(board, point) {
            policy[point.to_packed_index()] = 0.0;
        }
    }

    if policy_checker.is_policy_candidate(board, Point::default()) {
        policy[361] = 0.0;
    }

    // remove any symmetric moves that does not contribute to the search.
    //
    // we do this by finding all symmetries which provides symmetric board positions,
    // then for each candidate move we find the minimum index provided by some
    // symmetry.
    let symmetries = symmetry::ALL.iter()
        .filter(|&t| symmetry::is_symmetric(board, *t))
        .collect::<Vec<_>>();
    let mut indices = vec! [0; 362];
    indices[361] = 361;

    for point in Point::all() {
        let i = point.to_packed_index();

        if let Some(target) = symmetries.iter().map(|t| t.apply(point).to_packed_index()).min() {
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
    for point in Point::all() {
        let i = point.to_packed_index();
        let j = indices[t.inverse().apply(point).to_packed_index()];

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
        debug_assert!(!policy[i].is_nan(), "found NaN at index {}, total sum = {}", i, policy_sum);
    }
}

/// The shared variables between the master and each worker thread in the `predict` function.
#[derive(Clone)]
struct ThreadContext<T: TimeStrategy + Clone + Send> {
    /// The root of the monte carlo tree.
    root: Arc<UnsafeCell<tree::Node>>,

    /// The search options to use.
    options: Arc<Box<dyn SearchOptions>>,

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
fn predict_worker<T, P>(context: ThreadContext<T>, server: P)
    where T: TimeStrategy + Clone + Send + 'static,
          P: Predictor
{
    let root = unsafe { &mut *context.root.get() };
    let options = &**context.options;

    global_rwlock::read_lock();
    while !time_control::is_done(root, &context.time_strategy) {
        loop {
            let mut board = context.starting_point.clone();
            let trace = unsafe { tree::probe(root, &mut board) };
            global_rwlock::read_unlock();

            match trace {
                ProbeResult::Found(trace) => {
                    let &(_, color, _) = trace.last().unwrap();
                    let to_move = color.opposite();
                    let result = forward(&server, options, &board, to_move);

                    if let Some((value, policy)) = result {
                        global_rwlock::read_lock();

                        unsafe {
                            tree::insert(&trace, to_move, value, policy);
                            server.wake();
                            break
                        }
                    } else {
                        unsafe { tree::undo(trace, true) };

                        return  // unrecognized error
                    }
                },
                ProbeResult::Conflict => {
                    server.synchronize();
                    global_rwlock::read_lock();
                },
                ProbeResult::NoResult => {
                    return;
                }
            }
        }
    }
    global_rwlock::read_unlock();
}

/// Predicts the _best_ next move according to the given neural network when applied
/// to a monte carlo tree search.
///
/// # Arguments
///
/// * `server` - the server to use during evaluation
/// * `options` -
/// * `time_control` -
/// * `starting_tree` -
/// * `starting_point` -
/// * `starting_color` -
///
fn predict_aux<T, P>(
    server: &P,
    options: Box<dyn SearchOptions>,
    time_strategy: T,
    starting_tree: Option<tree::Node>,
    starting_point: &Board,
    starting_color: Color
) -> Option<(f32, usize, tree::Node)>
    where T: TimeStrategy + Clone + Send + 'static,
          P: Predictor + 'static
{
    let (starting_value, mut starting_policy) = full_forward::<P>(server, &*options, starting_point, starting_color)?;
    let deterministic = options.deterministic();

    // add some dirichlet noise to the root node of the search tree in order to increase
    // the entropy of the search and avoid overfitting to the prior value
    if !deterministic {
        dirichlet::add(&mut starting_policy[..362], 0.03);
    }

    // if we have a starting tree given, then re-use that tree (after some sanity
    // checks), otherwise we need to query the neural network about what the
    // prior value should be at the root node.
    let starting_tree = if let Some(mut starting_tree) = starting_tree {
        assert_eq!(starting_tree.to_move, starting_color);

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
    let num_workers = options.num_workers();
    let context = ThreadContext {
        root: Arc::new(UnsafeCell::new(starting_tree)),
        options: Arc::new(options),
        starting_point: starting_point.clone(),
        time_strategy: time_strategy.clone()
    };

    if num_workers <= 1 {
        let context = context.clone();
        let server = server.clone();

        predict_worker(context, server);
    } else {
        let handles = (0..num_workers).map(|_| {
            let context = context.clone();
            let server = server.clone();

            thread::Builder::new()
                .name("predict_worker".into())
                .spawn(move || predict_worker(context, server))
                .unwrap()
        }).collect::<Vec<JoinHandle<()>>>();

        // wait for all threads to terminate to avoid any zombie processes
        for handle in handles.into_iter() { handle.join().unwrap(); }
    }

    assert_eq!(Arc::strong_count(&context.root), 1);

    // choose the best move according to the search tree
    let root = UnsafeCell::into_inner(Arc::try_unwrap(context.root).ok().expect("no root"));
    let (value, index) = root.best(if !deterministic && starting_point.count() < 8 {
        *config::TEMPERATURE
    } else {
        0.0
    });

    #[cfg(feature = "trace-mcts")]
    eprintln!("{}", tree::to_sgf::<dg_go::utils::sgf::CGoban>(&root, starting_point, true));

    Some((value, index, root))
}

/// Predicts the _best_ next move according to the given neural network when applied
/// to a monte carlo tree search.
///
/// # Arguments
///
/// * `server` - the server to use during evaluation
/// * `options` -
/// * `time_control` -
/// * `starting_tree` -
/// * `starting_point` -
/// * `starting_color` -
///
pub fn predict<T, P>(
    server: &P,
    options: Box<dyn SearchOptions>,
    time_control: T,
    starting_tree: Option<tree::Node>,
    starting_point: &Board,
    starting_color: Color
) -> Option<(f32, usize, tree::Node)>
    where T: TimeStrategy + Clone + Send + 'static,
          P: Predictor + 'static
{
    predict_aux::<T, _>(server, options, time_control, starting_tree, starting_point, starting_color)
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
        let value: i32 = thread_rng().gen_range(-8, 8);

        value as f32 + 0.5
    }
}

#[cfg(test)]
mod tests {
    use dg_go::{Board, Color};
    use dg_utils::types::f16;
    use super::*;

    use std::sync::Arc;
    use std::cell::UnsafeCell;
    use options::StandardSearch;

    #[test]
    fn valid_komi() {
        // i do not like the use of randomness in tests, but I do not see much
        // choice here
        for _ in 0..10000 {
            let komi = get_random_komi();

            assert!(komi >= -7.5 && komi <= 7.5, "komi is {}", komi);
        }
    }

    #[test]
    fn no_allowed_moves() {
        let root = Arc::new(UnsafeCell::new(tree::Node::new(Color::Black, 0.0, vec! [1.0; 362])));
        let context = ThreadContext {
            root: root.clone(),
            starting_point: Board::new(7.5),
            options: Arc::new(Box::new(StandardSearch::new(1))),
            time_strategy: time_control::RolloutLimit::new(100)
        };

        for i in 0..362 {
            unsafe { &mut *context.root.get() }.disqualify(i);
        }

        predict_worker::<_, _>(context, predict::RandomPredictor::default());
        assert_eq!(unsafe { &*root.get() }.best(0.0), (::std::f32::NEG_INFINITY, 361));
    }

    #[derive(Clone, Default)]
    struct NanPredictor;

    impl predict::Predictor for NanPredictor {
        fn predict(&self, _features: Vec<f16>) -> Option<(f32, Vec<f32>)> {
            Some((0.0, vec! [::std::f32::NEG_INFINITY; 362]))
        }

        fn predict_all<E: Iterator<Item=Vec<f16>>>(&self, features_list: E) -> Vec<Option<(f32, Vec<f32>)>> {
            features_list.map(|features| self.predict(features)).collect()
        }

        fn synchronize(&self) {
            // pass
        }
    }

    #[test]
    fn no_finite_candidates() {
        let (value, index, root) = predict::<_, _>(
            &NanPredictor::default(),
            Box::new(StandardSearch::new(1)),
            time_control::RolloutLimit::new(1600),
            None,
            &Board::new(7.5),
            Color::Black
        ).unwrap();

        assert_eq!(value, 0.5);
        assert_eq!(index, 361);
        assert_eq!(root.total_count, 0);
        assert_eq!(root.vtotal_count, 0);
    }
}
