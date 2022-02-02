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
use crate::tree::{self, ProbeResult};
use crate::parallel::global_rwlock;
use super::event::{Event, EventKind};
use super::policy_helper::*;
use super::shared_context::{SharedContext, SearchContext};

use std::sync::atomic::Ordering;
use std::sync::{Arc, Barrier, RwLock};
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
    pub fn new(shared_context: Arc<SharedContext>, has_started: Arc<Barrier>) -> Self {
        shared_context.num_running.fetch_add(1, Ordering::AcqRel);
        has_started.wait();

        Self { shared_context }
    }

    pub fn run(&self, searches: Arc<RwLock<Vec<Arc<SearchContext>>>>) {
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
                            index = 0;
                            thread::yield_now();
                        }
                        TryProbeResult::Done { to_remove } => {
                            let mut searches_guard = searches.write().expect("could not acquire write lock");

                            if let Some(index) = searches_guard.iter().position(|search_context| search_context.id == to_remove) {
                                let search_context = searches_guard.swap_remove(index);
                                drop(searches_guard);

                                match search_context.response_channel.send(()) {
                                    _ => {}  // this is ok to fail
                                }
                            } else {
                                // someone else got here before us :-(
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
                    add_valid_candidates(&mut policy, &response.policy(), &indices, event.transformation);
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
        searches: &Arc<RwLock<Vec<Arc<SearchContext>>>>,
        mut index: usize
    ) -> TryProbeResult
    {
        let predictor = &self.shared_context.predictor;

        loop {
            // evaluate anything in the queue so far
            let event_responses = self.shared_context.batcher
                .get_batch(1)
                .map(|batch| batch.forward(predictor));

            if let Some((events, responses)) = event_responses {
                let event_queue = &self.shared_context.event_queue;
                for (event, response) in events.into_iter().zip(responses.into_iter()) {
                    event_queue.push(event.into_insert(response).1).ok().expect("could not push to event queue");
                }
            }

            // try to probe for something new
            let searches = searches.read().expect("could not acquire read lock");

            if let Some(search_context) = searches.get(index).cloned() {
                drop(searches);

                let root = unsafe { &mut *search_context.root };
                if global_rwlock::read(|| { time_control::is_done(root, &search_context.time_strategy) }) {
                    return TryProbeResult::Done { to_remove: search_context.id };
                }

                // probe the board if there has been an update since we last encountered
                // a conflict (or more than 1 ms has passed for deadlock reasons).
                let mut board = search_context.starting_point.clone();
                let probe = unsafe { global_rwlock::read(|| { tree::probe(root, &mut board) }) };

                return match probe {
                    ProbeResult::Found(trace) => {
                        self.shared_context.event_queue.push(Event::predict(predictor, search_context, board, trace)).ok().expect("could not push to event queue");
                        TryProbeResult::Retry { next_index: index + 1 }
                    },
                    ProbeResult::Conflict => {
                        TryProbeResult::Retry { next_index: index + 1 }
                    },
                    ProbeResult::NoResult => {
                        TryProbeResult::Done { to_remove: search_context.id }
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
