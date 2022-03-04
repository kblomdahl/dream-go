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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::{Predictor, Prediction};
use super::event::Event;
use dg_go::utils::features;
use dg_utils::config;
use dg_utils::types::f16;

pub struct Batch<'a> {
    hidden_states: Vec<f16>,
    features: Vec<f16>,
    events: Vec<Event>,
    num_batches: &'a AtomicUsize
}

impl<'a> Batch<'a> {
    pub fn new(hidden_states: Vec<f16>, features: Vec<f16>, events: Vec<Event>, num_batches: &'a AtomicUsize) -> Self {
        Self { hidden_states, features, events, num_batches }
    }

    pub fn forward(self, server: &Box<dyn Predictor + Sync>) -> (Vec<Event>, Vec<Prediction>) {
        let responses = server.predict(&self.hidden_states, &self.features, self.events.len());
        self.num_batches.fetch_sub(1, Ordering::AcqRel);

        (self.events, responses)
    }
}

pub struct BatcherList {
    /// The features gathered so far.
    features: Vec<f16>,

    /// The hidden states gathered so far.
    hidden_states: Vec<f16>,

    /// The events gathered so far.
    events: Vec<Event>,
}

impl BatcherList {
    fn new(max_batch_size: usize) -> Self {
        Self {
            features: Vec::with_capacity(2 * max_batch_size * features::Default::size()),
            hidden_states: Vec::with_capacity(2 * max_batch_size * 722),
            events: Vec::with_capacity(2 * max_batch_size)
        }
    }
}

#[derive(Clone)]
pub struct Batcher {
    /// The list of features and events gathered so far.
    list: Arc<Mutex<BatcherList>>,

    /// The number of batches "alive".
    num_batches: Arc<AtomicUsize>,

    /// The maximum size of a batch.
    max_batch_size: usize,

    /// The maximum number of allowed batches to be live at the same time.
    max_batches: usize,
}

impl Batcher {
    pub fn new(max_batches: usize) -> Self {
        let max_batch_size = *config::BATCH_SIZE;

        Self {
            list: Arc::new(Mutex::new(BatcherList::new(max_batch_size))),
            num_batches: Arc::new(AtomicUsize::new(0)),
            max_batch_size: max_batch_size,
            max_batches: max_batches
        }
    }

    pub fn push(&self, event: Event, hidden_states: Vec<f16>, features: Vec<f16>) {
        let mut list = self.list.lock().expect("could not acquire batch list lock");
        list.hidden_states.extend_from_slice(&hidden_states);
        list.features.extend_from_slice(&features);
        list.events.push(event);
    }

    pub fn push_and_get_batch(&self, event: Event, hidden_states: Vec<f16>, features: Vec<f16>) -> Option<Batch> {
        self.push(event, hidden_states, features);
        self.get_batch(self.max_batch_size)
    }

    pub fn get_batch(&self, min_batch_size: usize) -> Option<Batch> {
        // check so that we're not at capacity already
        let current = self.num_batches.load(Ordering::Acquire);

        if current >= self.max_batches {
            None
        } else {
            // check so that we're not returning a batch if we've already reached the threshold
            let mut list = self.list.lock().expect("could not acquire batch list lock");
            let size = list.events.len();

            if size >= min_batch_size && self.num_batches.compare_exchange_weak(current, current + 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                let split_index = if size >= self.max_batch_size { size - self.max_batch_size } else { 0 };

                Some(
                    Batch::new(
                        list.hidden_states.split_off(split_index * 722),
                        list.features.split_off(split_index * features::Default::size()),
                        list.events.split_off(split_index),
                        self.num_batches.as_ref()
                    )
                )
            } else {
                None
            }
        }
    }
}
