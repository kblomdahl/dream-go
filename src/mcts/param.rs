// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
use std::fmt;
use std::str::FromStr;

fn env_or_default<T: FromStr>(name: &str, default: T) -> T
    where <T as FromStr>::Err : fmt::Debug
{
    match env::var(name) {
        Ok(value) => value.parse::<T>()
                          .expect(&format!("{}: unexpected type -- {}", name, value)),
        Err(_) => default
    }
}

pub trait Param {
    /// The number of probes into the monte carlo tree to perform at each step. This
    /// integer must be dividable by both `thread_count` and `batch_size` and larger
    /// than zero.
    fn iteration_limit() -> usize;

    /// The number of threads to run in parallel probing into the search tree.
    fn thread_count() -> usize;

    /// The number of asynchronous neural network evaluations to batch together.
    fn batch_size() -> usize;

    /// How much dirichlet noise to add to the policy at the root of the
    /// monte carlo tree search.
    fn dirichlet_noise() -> f32;

    /// The exploration rate constant in the UCT formula, a higher value
    /// indicate a higher level of exploration.
    fn exploration_rate() -> f32;

    /// The absolute bias of the AMAF value compared to the local value,
    /// this can be heuristically determined based on what the difference
    /// between the final value and the AMAF value is.
    fn rave_bias() -> f32;

    /// The temperature of the opening, a larger temperature means the opening will
    /// be more random while a smaller value it will be greedy with respect to the
    /// moves suggested by the tree search.
    /// 
    /// This is applied to the first 8 moves of a game to ensure exploration and
    /// diversity. Past the first 8 moves, once we are in the middle game, it is
    /// too important to not make a tactical blunder that the temperature is forced
    /// to zero.
    fn temperature() -> f32;

    /// Whether to use experimental features. This is mainly used during internal
    /// testing of *new* features.
    fn experimental() -> bool;
}

#[derive(Clone)]
pub struct Standard;

impl Param for Standard {
    fn iteration_limit() -> usize {
        lazy_static! {
            static ref LIMIT: usize = {
                let limit = env_or_default("NUM_ITER", 800);

                assert!(limit > 0,
                    "The number of iterations ({}) must be larger than zero",
                    limit
                );

                limit
            };
        }

        *LIMIT
    }

    fn thread_count() -> usize {
        lazy_static! {
            static ref COUNT: usize = {
                let count = env_or_default("NUM_THREADS", 64);

                assert!(count > 0,
                    "The number of threads ({}) must be larger than zero",
                    count
                );

                count
            };
        }

        *COUNT
    }

    fn batch_size() -> usize {
        lazy_static! {
            static ref SIZE: usize = {
                let size = env_or_default("BATCH_SIZE", 16);

                assert!(size > 0,
                    "The batch size ({}) must be larger than zero",
                    size
                );

                size
            };
        }

        *SIZE
    }

    #[inline] fn dirichlet_noise() -> f32 { 0.25 }

    #[inline] fn exploration_rate() -> f32 {
        lazy_static! {
            static ref EXPLORATION_RATE: f32 = {
                let rate = env_or_default("EXPLORATION_RATE", 1.283165);

                assert!(rate >= 0.0,
                    "The exploration rate ({}) must be at least zero",
                    rate
                );

                rate
            };
        }

        *EXPLORATION_RATE
    }

    #[inline] fn rave_bias() -> f32 {
        lazy_static! {
            static ref RAVE_BIAS: f32 = {
                let bias = env_or_default("RAVE_BIAS", 0.705811);

                assert!(bias >= 0.0,
                    "The RAVE bias ({}) must be at least zero",
                    bias
                );

                bias
            };
        }

        *RAVE_BIAS
    }

    #[inline] fn temperature() -> f32 {
        lazy_static! {
            static ref TEMPERATURE: f32 = {
                let temp = env_or_default("TEMPERATURE", 0.7);

                assert!(temp > 0.0,
                    "The temperature ({}) must be at least zero",
                    temp
                );

                temp
            };
        }

        *TEMPERATURE
    }

    #[inline] fn experimental() -> bool { false }
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct Tournament;

impl Param for Tournament {
    #[inline] fn iteration_limit() -> usize { Standard::iteration_limit() }
    #[inline] fn thread_count() -> usize { Standard::thread_count() }
    #[inline] fn batch_size() -> usize { Standard::batch_size() }
    #[inline] fn dirichlet_noise() -> f32 { 0.1 }
    #[inline] fn exploration_rate() -> f32 { Standard::exploration_rate() }
    #[inline] fn rave_bias() -> f32 { Standard::rave_bias() }
    #[inline] fn experimental() -> bool { Standard::experimental() }

    #[inline] fn temperature() -> f32 {
        lazy_static! {
            static ref TEMPERATURE: f32 = {
                let temp = env_or_default("TEMPERATURE", 0.3);

                assert!(temp > 0.0,
                    "The temperature ({}) must be larger than zero",
                    temp
                );

                temp
            };
        }

        *TEMPERATURE
    }
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct Experimental;

impl Param for Experimental {
    #[inline] fn iteration_limit() -> usize { Standard::iteration_limit() }
    #[inline] fn thread_count() -> usize { Standard::thread_count() }
    #[inline] fn batch_size() -> usize { Standard::batch_size() }
    #[inline] fn dirichlet_noise() -> f32 { Standard::dirichlet_noise() }
    #[inline] fn exploration_rate() -> f32 { Standard::exploration_rate() }
    #[inline] fn rave_bias() -> f32 { Standard::rave_bias() }
    #[inline] fn temperature() -> f32 { Standard::temperature() }
    #[inline] fn experimental() -> bool { true }
}
