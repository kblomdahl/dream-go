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

use predict::{Predictor, PredictResponse};
use lru_cache::LruCache;
use dg_go::utils::symmetry::Transform;
use dg_go::{Board, Color};
use dg_cuda::Device;
use dg_nn::{self as nn, Network};
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
    to_move: Color,
    symmetry: Transform
}

impl BoardTuple {
    fn new(board: &Board, to_move: Color, symmetry: Transform) -> Self {
        Self {
            board_hash: board.zobrist_hash(),
            to_move: to_move,
            symmetry: symmetry
        }
    }
}

#[derive(Clone)]
pub struct PredictService {
    cache_table: Arc<Mutex<LruCache<BoardTuple, PredictResponse>>>,
    network: Network,
    count: Arc<AtomicUsize>
}

impl PredictService {
    pub fn new(network: Network) -> Self {
        let cache_table = Arc::new(Mutex::new(LruCache::with_capacity(MAX_CACHE_SIZE + 1)));
        let count = Arc::new(AtomicUsize::new(0));

        Self { cache_table, network, count }
    }
}

impl Predictor for PredictService {
    fn max_num_threads(&self) -> usize {
        let num_devices = Device::all().expect("could not find any compatible devices").len();
        2 * num_devices
    }

    fn fetch(&self, board: &Board, to_move: Color, symmetry: Transform) -> Option<PredictResponse> {
        let key = BoardTuple::new(board, to_move, symmetry);

        self.cache_table.lock().expect("could not acquire cache table lock")
            .get(&key)
            .cloned()
    }

    fn cache(&self, board: &Board, to_move: Color, symmetry: Transform, response: PredictResponse) {
        let key = BoardTuple::new(board, to_move, symmetry);

        self.cache_table.lock().expect("could not acquire cache table lock")
            .insert(&key, response);
    }

    fn predict(&self, features_list: &[f16], batch_size: usize) -> Vec<PredictResponse> {
        assert!(batch_size > 0);

        let devices = Device::all().expect("could not find any compatible devices");
        let index = self.count.fetch_add(1, Ordering::Relaxed) % devices.len();
        devices[index].set_current().expect("could not set the device for the current thread");

        //
        let network = &self.network;
        let result = network.get_workspace(batch_size).and_then(|mut workspace| {
            let outputs = nn::forward(&mut workspace, features_list)?;
            let (value_list, policy_list) = outputs.unwrap();
            let policy_iter = policy_list.chunks(362).map(|p| p.to_vec());

            Ok(
                value_list
                    .into_iter()
                    .zip(policy_iter)
                    .map(|(value, policy)| PredictResponse::new(value, policy))
                    .collect()
            )
        });

        result.expect("could not run neural network")
    }
}
