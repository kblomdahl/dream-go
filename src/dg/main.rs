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
#![feature(test)]

extern crate cpu_time;
extern crate dg_go;
extern crate dg_mcts;
extern crate dg_predict;
extern crate dg_sgf;
extern crate dg_utils;
#[macro_use] extern crate lazy_static;
extern crate regex;
#[cfg(test)] extern crate test;

mod bench;
mod gtp;

use dg_utils::config::{self, Procedure};

/// Main function.
fn main() {
    match &*config::PROCEDURE {
        Procedure::Help => {
            println!("Usage: ./dream-go [options]");
            println!();
            println!("  --self-play <n>          Extract a dataset from self-play containing n examples");
            println!("  --reanalyze <sgf...>     Extract a dataset by reanalyzing examples in the provided SGF");
            println!("                           files");
            println!("  --ex-it                  When combined with --policy-play perform search on some partial");
            println!("                           policies");
            println!("  --bench <sgf...>         Run benchmarks on the board positions in the provided SGF");
            println!("  --gtp                    Run GTP client (default)");
            println!();
            println!("Advanced options:");
            println!("  --safe-time <n>          The minimum number of milliseconds to leave on the game clock");
            println!("  --num-rollout <n>        The number of rollouts to add to the search tree for every move");
            println!("  --num-ex-it-rollout <n>  The number of rollouts to add to the search tree for every move");
            println!("  --num-games <n>          The number of games to play or extract in parallel");
            println!("  --num-threads <n>        The number of search threads to use in total");
            println!("  --num-samples <n>        The number of games to extract from each game record");
            println!("  --batch-size <n>         The number parallel rollouts to perform on the GPU");
            println!("  --tt                     Play using Tromp-Taylor rules");
            println!("  --no-ponder              Do not think in the background during idle time");
            println!("  --no-resign              Do not allow the engine to resign in games");
        },

        Procedure::Benchmark => {
            for sgf_file in config::get_args() {
                println!("{}:", sgf_file);
                println!("  sgf:       {:.4} per second", bench::SgfBenchmark::new().evaluate(&sgf_file));
                println!("  feature:   {:.4} per second", bench::FeatureBenchmark::new().evaluate(&sgf_file));
                println!("  batch_size {}", *config::BATCH_SIZE);
                println!("    forward: {:.4} per second", bench::ForwardBenchmark::new().evaluate(&sgf_file));
                println!("    mcts:    {:.4} per second", bench::MctsBenchmark::new().evaluate(&sgf_file));
            }
        },

        Procedure::Reanalyze(files) => {
            let (receiver, _server) = dg_mcts::reanalyze(files);

            for result in receiver.iter() {
                eprint!(".");
                println!("{}", result);
            }
        },

        Procedure::SelfPlay(n, ex_it) => {
            let (receiver, _server) = dg_mcts::self_play(*n, *ex_it);

            for result in receiver.iter() {
                eprint!(".");
                println!("{}", result);
            }
        },

        Procedure::Gtp => {
            gtp::run()
        }
    }
}
