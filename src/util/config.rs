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

use regex::Regex;

#[derive(PartialEq)]
pub enum Procedure {
    SelfPlay(usize),
    PolicyPlay(usize, bool),
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
    } else if has_opt("--policy-play") {
        Procedure::PolicyPlay(get_opt("--policy-play").unwrap_or(::std::usize::MAX), has_opt("--ex-it"))
    } else if has_opt("--self-play") {
        Procedure::SelfPlay(get_opt("--self-play").unwrap_or(1))
    } else {
        Procedure::Gtp
    };

    /// Whether to include Sabaki extentions amongst the GTP commands.
    pub static ref WITH_SABAKI: bool = has_opt("--with-sabaki");

    /// Whether to think in the background during idle time.
    pub static ref NO_PONDER: bool = has_opt("--no-ponder");

    /// The target number of rollouts for each search tree.
    pub static ref NUM_ROLLOUT: usize = get_opt("--num-rollout").unwrap_or(1600);

    /// The maximum batch size to forward to the neural network. A larger batch
    /// size typically result in a faster program but requires more GPU memory.
    pub static ref BATCH_SIZE: usize = get_opt("--batch-size").unwrap_or(32);

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

    /// Whether to output extra information for all actions.
    pub static ref VERBOSE: bool = has_opt("--verbose");

    /// The number of rollout to perform for each board position when playing
    /// _according to the policy_.
    pub static ref NUM_POLICY_ROLLOUT: usize = get_env("POLICY_ROLLOUT").unwrap_or(1);

    /// The amount of dirtchlet noise to add to the root node of each search
    /// tree. A larger value will result in a more random search, which is
    /// typically desirable during training but not during tournament play.
    pub static ref DIRICHLET_NOISE: f32 = get_env("DIRICHLET_NOISE")
        .unwrap_or_else(|| if *PROCEDURE == Procedure::Gtp { 0.05 } else { 0.25 });

    /// The temperature of the move selection during the eight first moves. A
    /// larger values make the engine more likely to pick a sub-optimal
    /// move (according to the search).
    pub static ref TEMPERATURE: f32 = get_env("TEMPERATURE")
        .unwrap_or_else(|| if *PROCEDURE == Procedure::Gtp { 0.3 } else { 0.92 });

    /// The softmax temperature to use at the end of the _policy head_. This
    /// temperature is applied for the entire game.
    pub static ref SOFTMAX_TEMPERATURE: f32 = get_env("SOFTMAX_TEMPERATURE")
        .unwrap_or(1.0);

    /// The _First Play Urgency_ reduction. Setting this is `1.0`, or `0.0`
    /// effectively disables FPU.
    pub static ref FPU_REDUCE: Vec<(i32, f32)> = get_intp_list("FPU_REDUCE")
        .unwrap_or(vec! [(0, 0.60), (200, 0.50), (800, 0.22)]);

    /// The number of virtual losses to add during async probes into the monte
    /// carlo search tree. A higher value avoids multiple probes exploring the
    /// same search tree.
    pub static ref VLOSS_CNT: i32 = get_env("VLOSS_CNT").unwrap_or(2);

    /// The UCT exploration rate.
    pub static ref UCT_EXP: Vec<(i32, f32)> = get_intp_list("UCT_EXP")
        .unwrap_or(vec! [(0, 0.88)]);
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

pub fn get_intp_list(name: &str) -> Option<Vec<(i32, f32)>> {
    lazy_static! {
        static ref POINT: Regex = Regex::new(r"([0-9]+),(0-9\.):?").unwrap();
    }

    get_env::<String>(name).and_then(|s| {
        let mut out: Vec<(i32, f32)> = POINT.captures_iter(&s).map(|point| {
            let x = point[1].parse::<i32>();
            let y = point[2].parse::<f32>();

            (x.unwrap(), y.unwrap())
        }).collect();

        out.sort_by_key(|p| { p.0 });

        if out.is_empty() {
            s.parse::<f32>().ok().map(|v| vec! [(0, v)])
        } else {
            Some(out)
        }
    })
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

fn get_intp_value(points: &Vec<(i32, f32)>, x: i32) -> f32 {
    if let Some(i) = points.iter().position(|e| e.0 >= x) {
        let x0 = points.get(if i == 0 { 0 } else { i-1 }).unwrap_or(&points[0]);
        let x1 = &points[i];
        let a = if x0.0 >= x1.0 {
            0.5
        } else {
            assert!(x1.0 >= x0.0);

            (x - x0.0) as f32 / (x1.0 - x0.0) as f32
        };

        (1.0 - a) * x0.1 + a * x1.1
    } else {
        points.last().unwrap().1
    }
}

/// Returns the UCT exploration constant as a function of the number of
/// visits to the **current** node.
///
/// # Arguments
///
/// * `visits` - 
///
pub fn get_uct_exp(visits: i32) -> f32 {
    get_intp_value(&UCT_EXP, visits)
}

/// Returns the first-play urgency constant as a function of the number of
/// visits to the **current** node.
///
/// # Arguments
///
/// * `visits` - 
///
pub fn get_fpu_reduce(visits: i32) -> f32 {
    get_intp_value(&FPU_REDUCE, visits)
}

#[cfg(test)]
mod tests {
    use util::config::*;

    #[test]
    fn intp_out_of_bounds_1() {
        assert_eq!(get_intp_value(&vec! [(0, 0.0), (100, 1.0)], -100), 0.0);
    }

    #[test]
    fn intp_out_of_bounds_2() {
        assert_eq!(get_intp_value(&vec! [(0, 0.0), (100, 1.0)], 200), 1.0);
    }

    #[test]
    fn intp_lower_edge() {
        assert_eq!(get_intp_value(&vec! [(0, 0.0), (100, 1.0)], 0), 0.0);
    }

    #[test]
    fn intp_upper_edge() {
        assert_eq!(get_intp_value(&vec! [(0, 0.0), (100, 1.0)], 100), 1.0);
    }

    #[test]
    fn intp_mid() {
        assert_eq!(get_intp_value(&vec! [(0, 0.0), (100, 1.0)], 40), 0.4);
    }
}

