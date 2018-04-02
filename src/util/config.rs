// Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use std::env;
use std::str::FromStr;

#[derive(PartialEq)]
pub enum Procedure {
    Extract(bool),
    SelfPlay(usize),
    PolicyPlay(usize),
    Gtp,
    Help
}

pub enum SamplingStrategy {
    Percent(f32),
    Fixed(usize)
}

impl FromStr for SamplingStrategy {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        let s = s.trim();

        if s.ends_with("%") {
            let s = s.trim_right_matches("%");

            s.parse::<f32>()
                .map_err(|_| ())
                .map(|p| SamplingStrategy::Percent(p / 100.0))
        } else {
            s.parse::<usize>()
                .map_err(|_| ())
                .map(|f| SamplingStrategy::Fixed(f))
        }
    }
}

lazy_static! {
    /// The main producedure to run during this execution.
    pub static ref PROCEDURE: Procedure = if has_opt("--help") {
        Procedure::Help
    } else if has_opt("--extract") {
        Procedure::Extract(has_opt("--ex-it"))
    } else if has_opt("--policy-play") {
        Procedure::PolicyPlay(get_opt("--policy-play").unwrap_or(::std::usize::MAX))
    } else if has_opt("--self-play") {
        Procedure::SelfPlay(get_opt("--self-play").unwrap_or(1))
    } else {
        Procedure::Gtp
    };

    /// Whether to include Sabaki extentions amongst the GTP commands.
    pub static ref NO_SABAKI: bool = has_opt("--no-sabaki");

    /// The target number of rollouts for each search tree.
    pub static ref NUM_ROLLOUT: usize = get_opt("--num-rollout").unwrap_or(1600);

    /// The maximum batch size to forward to the neural network. A larger batch
    /// size typically result in a faster program but requires more GPU memory.
    pub static ref BATCH_SIZE: usize = get_opt("--batch-size").unwrap_or(16);

    /// The maximum number of games to play in parallel during `SelfPlay`,
    /// `PolicyPlay`, and `Extract` (with expert iteration).
    pub static ref NUM_GAMES: usize = get_opt("--num-games")
        .unwrap_or_else(|| if *PROCEDURE == Procedure::Gtp { 1 } else { 16 });

    /// The total number of parallel probes to perform for every monte carlo
    /// search tree.
    /// 
    /// When trying to improve the GPU utilization you should prefer to
    /// increase the `NUM_GAMES` variable instead as that scaled much better.
    pub static ref NUM_THREADS: usize = {
        let num_threads = get_opt("--num-threads").unwrap_or(64);

        assert!(
            num_threads >= *NUM_GAMES,
            "The number of threads must be at least the same as the number of games"
        );

        num_threads
    };

    /// The number of samples to extract from each game record.
    pub static ref NUM_SAMPLES: SamplingStrategy = get_opt("--num-samples")
        .unwrap_or(SamplingStrategy::Percent(0.01));

    /// The amount of dirtchlet noise to add to the root node of each search
    /// tree. A larger value will result in a more random search, which is
    /// typically desirable during training but not during tournament play.
    pub static ref DIRICHLET_NOISE: f32 = get_env("DIRICHLET_NOISE")
        .unwrap_or_else(|| if *PROCEDURE == Procedure::Gtp { 0.05 } else { 0.25 });

    /// The temperature of the move selection during the eight first moves. A
    /// larger values make the engine more likely to pick a sub-optimal
    /// move (according to the search).
    pub static ref TEMPERATURE: f32 = get_env("TEMPERATURE")
        .unwrap_or_else(|| if *PROCEDURE == Procedure::Gtp { 0.3 } else { 0.97 });

    /// The _First Play Urgency_ reduction. Setting this is `1.0` effectively
    /// disables FPU.
    pub static ref FPU_REDUCE: f32 = get_env("FPU_REDUCE").unwrap_or(0.22);

    /// The number of virtual losses to add during async probes into the monte
    /// carlo search tree. A higher value avoids multiple probes exploring the
    /// same search tree.
    pub static ref VLOSS_CNT: i32 = get_env("VLOSS_CNT").unwrap_or(1);

    /// The UCT exploration rate.
    pub static ref UCT_EXP: f32 = get_env("UCT_EXP")
        .unwrap_or(0.88);

    /// The rave bias.
    pub static ref RAVE_BIAS: f32 = get_env("RAVE_BIAS")
        .unwrap_or(0.705811);
}

/// Returns true if any command-line argument with the given name is present.
/// 
/// # Arguments
/// 
/// * `name` - the command-line arguments to check for
/// 
pub fn has_opt(name: &str) -> bool {
    env::args().skip(1).any(|arg| arg == name)
}

pub fn get_opt<T: FromStr>(name: &str) -> Option<T> {
    env::args().skip(1).zip(env::args().skip(2))
        .filter_map(|(arg, value)| {
            if arg == name {
                T::from_str(&value).ok()
            } else {
                None
            }
        })
        .next()
}

pub fn get_env<T: FromStr>(name: &str) -> Option<T> {
    match env::var(name) {
        Ok(value) => T::from_str(&value).ok(),
        _ => None
    }
}

/// Returns all unnamed arguments given to this program.
pub fn get_args() -> Vec<String> {
    let mut rest = vec! [];

    for (i, arg) in env::args().enumerate().skip(1) {
        if !arg.starts_with("--") && !usize::from_str(&arg).is_ok() {
            rest.push(arg);
        } else if arg == "--" {
            for arg in env::args().skip(i + 1) {
                rest.push(arg);
            }

            break
        }
    }

    rest
}
