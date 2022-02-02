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
use dg_go::utils::features::{self, HWC, Features};
use dg_go::utils::sgf::SgfEntry;
use dg_go::utils::symmetry::Transform;
use dg_utils::config;
use dg_utils::types::f16;
use dg_predict::{Builder, Model};

pub struct ForwardBenchmarkExecutor {
    batch_size: usize,
    model: Model
}

impl BenchmarkExecutor for ForwardBenchmarkExecutor {
    fn new() -> Self {
        let batch_size = *config::BATCH_SIZE;
        let model = Builder::default()
            .build()
            .expect("could not build predictor model");

        Self { batch_size, model }
    }

    fn call(&mut self, entry: SgfEntry) -> usize {
        let mut features = features::Default::new(&entry.board).get_features::<HWC, f16>(entry.color, Transform::Identity);
        if self.batch_size > 1 {
            features = features.repeat(self.batch_size);
        }

        let _out = self.model.initial_predict(&features, self.batch_size).unwrap();
        self.batch_size
    }
}

pub type ForwardBenchmark = Benchmark<ForwardBenchmarkExecutor>;
