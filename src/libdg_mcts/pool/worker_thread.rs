// Copyright 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use crate::time_control;
use crate::options::SearchOptions;
use crate::tree::{self, ProbeResult};
use crate::parallel::global_rwlock;
use super::event::{Event, EventKind};
use super::shared_context::{SharedContext, SearchContext};
use dg_go::utils::symmetry;
use dg_go::{Point, Board, Color};

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;

enum TryProbeResult {
    Done { to_remove: usize },
    Quit,
    Retry { next_index: usize }
}

pub struct Worker {
    shared_context: Arc<SharedContext>
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shared_context.num_running.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Worker {
    pub fn new(shared_context: Arc<SharedContext>) -> Self {
        shared_context.num_running.fetch_add(1, Ordering::AcqRel);

        Self { shared_context }
    }

    pub fn run(&self, mut searches: Vec<Arc<SearchContext>>) {
        let batcher = &self.shared_context.batcher;
        let event_queue = &self.shared_context.event_queue;
        let is_running = &self.shared_context.is_running;
        let predictor = &self.shared_context.predictor;
        let mut index = 0;

        'outer: loop {
            match event_queue.pop().map(|event| event.into_pending()).ok() {
                None => {
                    if !is_running.load(Ordering::Acquire) {
                        break 'outer;
                    }

                    match self.try_probe(&searches, index) {
                        TryProbeResult::Retry { next_index } => {
                            index = next_index
                        },
                        TryProbeResult::Quit => {
                            break 'outer;
                        }
                        TryProbeResult::Done { to_remove } => {
                            let index = searches.iter()
                                .position(|search_context| search_context.id == to_remove)
                                .expect("could not locate `search_context`");
                            let search_context = searches.swap_remove(index);

                            match search_context.response_channel.send(()) {
                                _ => {}  // this is ok to fail
                            }
                        }
                    }
                },
                Some((EventKind::Predict(features), event)) => {
                    // add to the end of the queue
                    let event_responses = batcher
                        .push_and_get_batch(event, features)
                        .map(|batch| batch.forward(predictor));

                    // if we got a batch back from the queue then evaluate it
                    if let Some((events, responses)) = event_responses {
                        for (event, response) in events.into_iter().zip(responses.into_iter()) {
                            event_queue.push(event.into_insert(response).1).ok().expect("could not push to event queue");
                        }
                    }
                },
                Some((EventKind::Insert(response), event)) => {
                    let options = &event.search_context.options;
                    let &(_, last_move, _) = event.trace.last().unwrap();
                    let to_move = last_move.opposite();
                    let (mut policy, indices) = create_initial_policy(options, &event.board, to_move);
                    add_valid_candidates(&mut policy, response.policy(), &indices, event.transformation);
                    normalize_policy(&mut policy, 1.0);

                    unsafe {
                        global_rwlock::read(|| { tree::insert(&event.trace, to_move, response.winrate(), policy) });
                        predictor.cache(&event.board, to_move, event.transformation, response);
                    }
                },
                Some((EventKind::Pending, _)) => {
                    unreachable!();
                }
            }
        }
    }

    fn try_probe(
        &self,
        searches: &Vec<Arc<SearchContext>>,
        mut index: usize
    ) -> TryProbeResult
    {
        let predictor = &self.shared_context.predictor;

        loop {
            if let Some(search_context) = searches.get(index) {
                // evaluate anything in the queue so far
                let root = unsafe { &mut *search_context.root };
                let event_responses = self.shared_context.batcher
                    .get_batch(1)
                    .map(|batch| batch.forward(predictor));

                if let Some((events, responses)) = event_responses {
                    let event_queue = &self.shared_context.event_queue;
                    for (event, response) in events.into_iter().zip(responses.into_iter()) {
                        event_queue.push(event.into_insert(response).1).ok().expect("could not push to event queue");
                    }

                    return TryProbeResult::Retry { next_index: index + 1 }
                } else {
                    if global_rwlock::read(|| { time_control::is_done(root, &search_context.time_strategy) }) {
                        return TryProbeResult::Done { to_remove: search_context.id };
                    }

                    // probe the board if there has been an update since we last encountered
                    // a conflict (or more than 1 ms has passed for deadlock reasons).
                    let mut board = search_context.starting_point.clone();
                    let probe = unsafe { global_rwlock::read(|| { tree::probe(root, &mut board) }) };

                    return match probe {
                        ProbeResult::Found(trace) => {
                            self.shared_context.event_queue.push(Event::predict(predictor, search_context.clone(), board, trace)).ok().expect("could not push to event queue");
                            TryProbeResult::Retry { next_index: index + 1 }
                        },
                        ProbeResult::Conflict => {
                            thread::yield_now();
                            TryProbeResult::Retry { next_index: index + 1 }
                        },
                        ProbeResult::NoResult => {
                            TryProbeResult::Done { to_remove: search_context.id }
                        }
                    }
                }
            } else if searches.is_empty() {
                return TryProbeResult::Quit;
            } else {
                index = (index + 1) % searches.len();  // retry with the new index
            }
        }
    }
}

/// Returns a initial accumulator policy where all illegal moves has been set
/// to _-Inf_, as well as an symmetry elimination mapping for its indices.
///
/// # Arguments
///
/// * `options` -
/// * `board` -
/// * `color` -
///
fn create_initial_policy(options: &Box<dyn SearchOptions + Sync>, board: &Board, to_move: Color) -> (Vec<f32>, Vec<usize>) {
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

/// Normalize the given vector so that its elements sums to `sum_to`.
///
/// # Arguments
///
/// * `policy` - the vector to normalize in-place
/// * `sum_to` - the value that the elements should sum to
///
fn normalize_policy(policy: &mut [f32], sum_to: f32) {
    use crate::asm::sum_finite_f32;
    use crate::asm::normalize_finite_f32;

    // re-normalize the policy since we have modified its values
    let policy_sum: f32 = sum_finite_f32(&policy);

    if policy_sum < 1e-6 {  // do not divide by zero
        let num_finite = policy.iter().filter(|x| x.is_finite()).count() as f32;

        for x in policy.iter_mut().filter(|x| x.is_finite()) {
            *x = sum_to / num_finite;
        }
    } else {
        normalize_finite_f32(policy, policy_sum / sum_to);
    }

    // check for NaN
    for i in 0..362 {
        debug_assert!(!policy[i].is_nan(), "found NaN at index {}, total sum = {}", i, policy_sum);
    }
}
