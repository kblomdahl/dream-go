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
extern crate concurrent_queue;
extern crate crossbeam_utils;
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
mod lru_cache;
mod greedy_score;
pub mod options;
pub mod parallel;
pub mod predictor;
pub mod predictors;
mod reanalyze;
mod self_play;
pub mod tree;
pub mod time_control;
pub mod pool;

/* -------- Exports -------- */

pub use self::game_result::*;
pub use self::greedy_score::*;
pub use self::self_play::*;
pub use self::reanalyze::*;

/* -------- Code -------- */

use rand::{thread_rng, Rng};
use std::cell::UnsafeCell;

use dg_go::utils::features::{self, HWC, Features};
use dg_go::utils::symmetry;
use dg_go::{Board, Color};
use self::options::{SearchOptions, ScoringSearch};
use self::time_control::TimeStrategy;
use self::tree::NodeTrace;
use self::predictor::{Predictor, Prediction};
use dg_utils::config;
use dg_utils::types::f16;
use self::pool::*;

/// Return the value and policy for the given board position, as the interpolation
/// of their value for every symmetry.
///
/// # Arguments
///
/// * `predictor` - the server to use for predictions
/// * `options` -
/// * `board` - the board position to evaluate
/// * `to_move` - the color to evaluate for
///
fn full_forward(predictor: &dyn Predictor, options: &Box<dyn SearchOptions + Sync>, board: &Board, to_move: Color) -> Option<(f32, Vec<f32>)> {
    let (initial_policy, indices) = create_initial_policy(options, board, to_move);
    let mut policy = initial_policy.clone();
    let mut value = 0.0f32;

    // find out which symmetries has already been calculated, and which ones has not
    let mut new_requests = Vec::with_capacity(8 * features::Default::size());
    let mut new_symmetries = Vec::with_capacity(8);

    for &t in &symmetry::ALL {
        if let Some(new_response) = predictor.fetch(board, to_move, t) {
            let mut new_policy = initial_policy.clone();
            add_valid_candidates(&mut new_policy, new_response.policy(), &indices, t);
            normalize_policy(&mut new_policy, 0.125);

            value += new_response.winrate() * 0.125;
            for i in 0..362 {
                policy[i] += new_policy[i];
            }
        } else {
            let features = features::Default::new(&board).get_features::<HWC, f16>(to_move, t);
            new_requests.extend_from_slice(&features);
            new_symmetries.push(t);
        }
    }

    // calculate any symmetries that were missing, add them to the cache, and then take the
    // average of them
    let batch_size = new_symmetries.len();

    if batch_size > 0 {
        let new_responses = predictor.predict(&new_requests, batch_size);

        for (new_response, t) in new_responses.into_iter().zip(new_symmetries.into_iter()) {
            let mut new_policy = initial_policy.clone();
            add_valid_candidates(&mut new_policy, new_response.policy(), &indices, t);
            normalize_policy(&mut new_policy, 0.125);

            value += new_response.winrate() * 0.125;
            for i in 0..362 {
                policy[i] += new_policy[i];
            }
            predictor.cache(board, to_move, t, new_response);
        }
    }

    Some((value, policy))
}

/// Predicts the _best_ next move according to the given neural network when applied
/// to a monte carlo tree search.
///
/// # Arguments
///
/// * `pool` - the worker pool to use for evaluation
/// * `options` -
/// * `time_control` -
/// * `starting_tree` -
/// * `starting_point` -
/// * `starting_color` -
///
pub fn predict(
    pool: &Pool,
    options: Box<dyn SearchOptions + Sync>,
    time_strategy: Box<dyn TimeStrategy + Sync>,
    starting_tree: Option<tree::Node>,
    starting_point: &Board,
    starting_color: Color
) -> Option<(f32, usize, tree::Node)>
{
    let deterministic = options.deterministic();
    let (starting_value, mut starting_policy) = full_forward(
        pool.predictor(),
        &options,
        starting_point,
        starting_color
    )?;

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

    // enqueue this tree search
    let root = UnsafeCell::new(starting_tree);
    pool.enqueue(root.get(), options, time_strategy, starting_point.clone())?;

    // choose the best move according to the search tree
    let root = UnsafeCell::into_inner(root);
    let (value, index) = root.best(if !deterministic && starting_point.count() < 8 {
        *config::TEMPERATURE
    } else {
        0.0
    });

    #[cfg(feature = "trace-mcts")]
    eprintln!("{}", tree::to_sgf::<dg_go::utils::sgf::CGoban>(&root, starting_point, true));

    Some((value, index, root))
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
        let value: i32 = thread_rng().gen_range(-8..8);

        value as f32 + 0.5
    }
}

#[cfg(test)]
mod tests {
    use dg_go::{Board, Color};
    use super::*;

    use options::StandardDeterministicSearch;
    use predictors::{RandomPredictor, NanPredictor};

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
        let pool = Pool::with_capacity(Box::new(RandomPredictor::default()), 1);
        let mut root = tree::Node::new(Color::Black, 0.0, vec! [1.0; 362]);

        for i in 0..362 {
            root.disqualify(i);
        }

        let (_value, _index, tree) = predict(
            &pool,
            Box::new(StandardDeterministicSearch::new()),
            Box::new(time_control::RolloutLimit::new(100)),
            Some(root),
            &Board::new(7.5),
            Color::Black
        ).expect("could not predict a position");

        assert_eq!(tree.best(0.0), (::std::f32::NEG_INFINITY, 361));
    }

    #[test]
    fn no_finite_candidates() {
        let (value, index, root) = predict(
            &Pool::with_capacity(Box::new(NanPredictor::default()), 1),
            Box::new(StandardDeterministicSearch::new()),
            Box::new(time_control::RolloutLimit::new(1600)),
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
