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

use bench::{Benchmark, BenchmarkExecutor};
use dg_go::utils::sgf::SgfEntry;
use dg_mcts::options::StandardSearch;
use dg_mcts::predict_service::PredictService;
use dg_mcts::pool::Pool;
use dg_mcts::predict;
use dg_mcts::time_control::RolloutLimit;
use dg_nn::Network;
use dg_utils::config;

pub struct MctsBenchmarkExecutor {
    pool: Pool
}

impl BenchmarkExecutor for MctsBenchmarkExecutor {
    fn new(network: Network) -> Self {
        let pool = Pool::new(Box::new(PredictService::new(network)));

        Self { pool }
    }

    fn call(&mut self, entry: SgfEntry) -> usize {
        let (_value, _, tree) = predict(
            &self.pool,
            Box::new(StandardSearch::default()),
            Box::new(RolloutLimit::new(usize::from(*config::NUM_ROLLOUT))),
            None,
            &entry.board,
            entry.color
        ).unwrap();

        tree.total_count as usize
    }
}

pub type MctsBenchmark = Benchmark<MctsBenchmarkExecutor>;
