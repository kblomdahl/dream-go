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

use dg_go::{Board, Color};
use dg_sgf::*;

use std::fs::File;
use std::time::Instant;
use std::io::{BufRead, BufReader};

pub trait BenchmarkExecutor {
    fn new() -> Self;
    fn setup(&mut self);
    fn call(&mut self, board: &Board, to_move: Color) -> usize;
}

pub struct Benchmark<B: BenchmarkExecutor> {
    executor: B
}

impl<B: BenchmarkExecutor> Benchmark<B> {
    pub fn new() -> Self {
        Self {
            executor: B::new()
        }
    }

    pub fn evaluate(&mut self, sgf_file: &str) -> f64 {
        let start_time = Instant::now();
        let mut count = 0;

        if let Ok(f) = File::open(sgf_file) {
            for line in BufReader::new(&f).lines().map(|x| x.unwrap()) {
                let mut board = Board::new(0.5);

                self.executor.setup();
                for tok in Stream::new(line.as_bytes()) {
                    let changed = match tok {
                        SgfToken::Komi { .. } => {
                            board = Board::new(tok.number());
                            true
                        },
                        SgfToken::Add { .. } | SgfToken::Play { .. } => {
                            let at_point = tok.point();
                            let to_move = tok.color();

                            if board.is_valid(to_move, at_point) {
                                board.place(to_move, at_point);
                                true
                            } else {
                                break
                            }
                        }
                        _ => false
                    };

                    if changed {
                        let to_move = board.last_played().map(|c| c.opposite()).unwrap_or(Color::Black);

                        count += self.executor.call(&board, to_move);
                    }
                }
            }
        }

        (count as f64) / start_time.elapsed().as_secs_f64()
    }
}
