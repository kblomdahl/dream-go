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

use dg_go::Board;
use crate::options::SearchOptions;
use crate::time_control::TimeStrategy;
use crate::tree;
use crate::predict::Predictor;
use super::batch::Batcher;
use super::event::Event;

use concurrent_queue::ConcurrentQueue;
use crossbeam_channel::Sender;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

///
pub struct SearchContext {
    pub id: usize,
    pub root: *mut tree::Node,
    pub options: Box<dyn SearchOptions + Sync>,
    pub time_strategy: Box<dyn TimeStrategy + Sync>,
    pub starting_point: Board,
    pub response_channel: Sender<()>
}

unsafe impl Send for SearchContext {}  // because of `UnsafeCell`
unsafe impl Sync for SearchContext {}  // because of `UnsafeCell`

impl SearchContext {
    pub fn new(
        id: usize,
        root: *mut tree::Node,
        options: Box<dyn SearchOptions + Sync>,
        time_strategy: Box<dyn TimeStrategy + Sync>,
        starting_point: Board,
        response_channel: Sender<()>
    ) -> Self
    {
        Self {
            id, root, options, time_strategy, starting_point, response_channel
        }
    }
}

/// The context that is shared between all of the workers in the pool. It
/// contains all of the information necessary for worker coordination and
/// termination.
pub struct SharedContext {
    pub is_running: AtomicBool,
    pub num_running: AtomicUsize,
    pub event_queue: ConcurrentQueue<Event>,
    pub predictor: Box<dyn Predictor + Sync>,
    pub batcher: Batcher,
}

impl SharedContext {
    pub fn new(predictor: Box<dyn Predictor + Sync>) -> Self {
        let max_num_threads = predictor.max_num_threads();

        Self {
            is_running: AtomicBool::new(true),
            num_running: AtomicUsize::new(0),
            event_queue: ConcurrentQueue::unbounded(),
            predictor: predictor,
            batcher: Batcher::new(max_num_threads)
        }
    }
}

impl Drop for SharedContext {
    fn drop(&mut self) {
        assert_eq!(self.is_running.load(Ordering::Acquire), false);
        assert_eq!(self.num_running.load(Ordering::Acquire), 0);
    }
}
