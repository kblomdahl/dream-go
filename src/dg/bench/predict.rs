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

use bench::{Benchmark, BenchmarkExecutor, PredictState1};
use dg_go::utils::features::{HWC, Features};
use dg_go::utils::sgf::SgfEntry;
use dg_go::utils::symmetry::Transform;
use dg_utils::config;
use dg_utils::types::f16;
use dg_nn::Network;
use dg_mcts::predict_service::{PredictState, PredictRequest};
use dg_mcts::parallel;

pub struct PredictBenchmarkExecutor {
    batch_size: usize,
    server: parallel::Service<PredictState1>
}

impl BenchmarkExecutor for PredictBenchmarkExecutor {
    fn new(network: Network) -> Self {
        let batch_size = *config::BATCH_SIZE;
        let _workspace1 = network.get_workspace(batch_size).expect("could not create `Workspace` from `Network`");
        let _workspace2 = network.get_workspace(batch_size).expect("could not create `Workspace` from `Network`");

        Self {
            batch_size: batch_size,
            server: parallel::Service::new(None, PredictState::new(network))
        }
    }

    fn call(&mut self, entry: SgfEntry) -> usize {
        let features = entry.board.get_features::<HWC, f16>(entry.color, Transform::Identity);
        let server_lock = self.server.lock();

        if self.batch_size > 1 {
            let batch = vec! [features].into_iter().cycle().take(self.batch_size);

            server_lock.send_all(batch.map(|x| PredictRequest::Ask(x))).unwrap();
        } else {
            server_lock.send(PredictRequest::Ask(features)).unwrap();
        }

        self.batch_size
    }    
}

pub type PredictBenchmark = Benchmark<PredictBenchmarkExecutor>;
