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

use predictor::{Predictor, Prediction};
use lru_cache::LruCache;
use dg_go::{Board, Color};
use dg_cuda::Device;
use dg_predict::{Builder, Model};
use dg_utils::types::f16;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

/// The maximum number of entries to be stored in the transposition table
/// before we need to remove the least recently used one.
const MAX_CACHE_SIZE: usize = 200_000;

#[derive(Clone, Hash, PartialEq, Eq)]
struct BoardTuple {
    board_hash: u64,
    to_move: Color
}

impl BoardTuple {
    fn new(board: &Board, to_move: Color) -> Self {
        Self {
            board_hash: board.zobrist_hash(),
            to_move: to_move
        }
    }
}

#[derive(Clone)]
pub struct NnPredictor {
    cache_table: Arc<Mutex<LruCache<BoardTuple, Prediction>>>,
    model: Arc<Model>,
    count: Arc<AtomicUsize>
}

impl Default for NnPredictor {
    fn default() -> Self {
        let model = Arc::new(
            Builder::default().build()
                .expect("could not build model")
        );
        let cache_table = Arc::new(Mutex::new(LruCache::with_capacity(MAX_CACHE_SIZE + 1)));
        let count = Arc::new(AtomicUsize::new(0));

        Self { cache_table, model, count }
    }
}

impl Predictor for NnPredictor {
    fn max_num_threads(&self) -> usize {
        let num_devices = Device::all().expect("could not find any compatible devices").len();
        2 * num_devices
    }

    fn fetch(&self, board: &Board, to_move: Color) -> Option<Prediction> {
        let key = BoardTuple::new(board, to_move);

        self.cache_table.lock().expect("could not acquire cache table lock")
            .get(&key)
            .cloned()
    }

    fn cache(&self, board: &Board, to_move: Color, response: Prediction) {
        let key = BoardTuple::new(board, to_move);

        self.cache_table.lock().expect("could not acquire cache table lock")
            .insert(&key, response.clone());
    }

    fn initial_predict(&self, features_list: &[f16], batch_size: usize) -> Vec<Prediction> {
        assert!(batch_size > 0);

        let devices = Device::all().expect("could not find any compatible devices");
        let index = self.count.fetch_add(1, Ordering::Relaxed) % devices.len();
        devices[index].set_current().expect("could not set the device for the current thread");

        //
        let model = &self.model;
        let outputs = model.initial_predict(features_list, batch_size)
            .expect("could not execute neural network");

        outputs.value.iter()
            .zip(outputs.policy.chunks_exact(362))
            .zip(outputs.hidden_states.chunks_exact(722))
            .map(|((&value, policy), hidden_states)| Prediction::new(value, policy.to_vec(), hidden_states.to_vec()))
            .collect()
    }

    fn predict(&self, hidden_features_list: &[f16], features_list: &[f16], batch_size: usize) -> Vec<Prediction> {
        assert!(batch_size > 0);

        let devices = Device::all().expect("could not find any compatible devices");
        let index = self.count.fetch_add(1, Ordering::Relaxed) % devices.len();
        devices[index].set_current().expect("could not set the device for the current thread");

        //
        let model = &self.model;
        let outputs = model.predict(hidden_features_list, features_list, batch_size)
            .expect("could not execute neural network");

        outputs.value.iter()
            .zip(outputs.policy.chunks_exact(362))
            .zip(outputs.hidden_states.chunks_exact(722))
            .map(|((&value, policy), hidden_states)| Prediction::new(value, policy.to_vec(), hidden_states.to_vec()))
            .collect()
    }
}
