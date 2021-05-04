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

use crate::predict::Predictor;
use crate::options::SearchOptions;
use crate::time_control::TimeStrategy;
use crate::tree;
use dg_go::Board;
use dg_utils::config;

use crossbeam_channel;
use crossbeam_utils::Backoff;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};

use super::shared_context::{SharedContext, SearchContext};
use super::worker_thread::Worker;

#[derive(Clone)]
pub struct Pool {
    shared_context: Arc<SharedContext>,
    searches_count: Arc<AtomicUsize>,
    searches: Arc<Mutex<Vec<Arc<SearchContext>>>>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>
}

impl Pool {
    pub fn new(predictor: Box<dyn Predictor + Sync>) -> Self {
        Self {
            shared_context: Arc::new(SharedContext::new(predictor)),
            searches_count: Arc::new(AtomicUsize::new(0)),
            searches: Arc::new(Mutex::new(Vec::with_capacity(8))),
            handles: Arc::new(Mutex::new(Vec::with_capacity(64)))
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared_context.is_running.store(false, Ordering::Release);

        for handle in self.handles.lock().expect("could not acquire lock").drain(..) {
            handle.join().expect("could not terminal worker thread");
        }
    }
}

impl Pool {
    fn ensure_threads(&self, searches: MutexGuard<Vec<Arc<SearchContext>>>) {
        let shared_context = self.shared_context.as_ref();
        let num_threads = *config::NUM_THREADS;
        let mut handles = self.handles.lock().expect("could not acquire lock");

        // join any existing threads
        shared_context.is_running.store(false, Ordering::Release);
        for handle in handles.drain(..) {
            handle.join().expect("could not join worker thread");
        }

        // start-up new threads with the latest `searches` list
        shared_context.is_running.store(true, Ordering::Release);
        while shared_context.is_running.load(Ordering::Acquire) && shared_context.num_running.load(Ordering::Acquire) < num_threads {
            let shared_context = self.shared_context.clone();
            let searches = searches.clone();

            handles.push(thread::spawn(move || Worker::new(shared_context).run(searches)));
        }
    }

    pub fn predictor(&self) -> &dyn Predictor {
        self.shared_context.predictor.as_ref()
    }

    pub fn enqueue(
        &self,
        root: *mut tree::Node,
        options: Box<dyn SearchOptions + Sync>,
        time_strategy: Box<dyn TimeStrategy + Sync>,
        starting_point: Board
    ) -> Option<()>
    {
        // add this board position to the worker pool, and **make sure** to drop
        // the write-lock :-)
        let (tx, rx) = crossbeam_channel::bounded(1);
        let next_id = self.searches_count.fetch_add(1, Ordering::AcqRel);
        let search_context = Arc::new(SearchContext::new(
                next_id,
                root,
                options,
                time_strategy,
                starting_point,
                tx
            )
        );

        let mut searches = self.searches.lock().expect("could not acquire lock");
        searches.push(search_context.clone());

        self.ensure_threads(searches);

        // wait for the worker pool to finish their work
        let result = rx.recv().ok();
        drop(rx);

        // remove the `search_context` from the queue and wait until everyone
        // has dropped it from their list.
        let backoff = Backoff::new();

        self.searches.lock()
            .expect("could not acquire lock")
            .retain(|search_context| search_context.id != next_id);

        while Arc::strong_count(&search_context) > 1 {
            backoff.snooze();
        }

        result
    }
}

#[cfg(test)]
mod tests {
    // pass
}
