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

use dg_go::utils::sgf::{self, Sgf, SgfEntry};
use dg_nn::Network;

use std::fs::File;
use std::time::Instant;
use std::io::{BufRead, BufReader};

pub trait BenchmarkExecutor {
    fn new(network: Network) -> Self;
    fn call(&mut self, entry: SgfEntry) -> usize;
}

pub struct Benchmark<B: BenchmarkExecutor> {
    executor: B
}

impl<B: BenchmarkExecutor> Benchmark<B> {
    pub fn new(network: &Network) -> Self {
        Self {
            executor: B::new(network.clone())
        }
    }

    pub fn evaluate(&mut self, sgf_file: &str) -> f64 {
        let start_time = Instant::now();
        let mut count = 0;

        if let Ok(f) = File::open(sgf_file) {
            for line in BufReader::new(&f).lines().map(|x| x.unwrap()) {
                if let Ok(komi) = sgf::get_komi_from_sgf(&line) {
                    for entry in Sgf::new(line.as_bytes(), komi) {
                        if let Ok(entry) = entry {
                            count += self.executor.call(entry);
                        }
                    }
                }
            }
        }

        (count as f64) / start_time.elapsed().as_secs_f64()
    }
}
