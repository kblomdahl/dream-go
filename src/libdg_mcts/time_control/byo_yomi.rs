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

use super::{TimeStrategy, TimeStrategyResult};
use tree;
use dg_utils::config::SAFE_TIME_MS;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The buffer time to remove from every period no matter what. This is to compensate for
/// any latency in the rest of the program.
const PERIOD_BUF_TIME_MS: usize = 50;

#[derive(Clone)]
pub struct ByoYomi {
    /// The total remaining time in milliseconds
    total_time_ms: usize,

    /// The number of visits the tree had in the beginning
    starting_visits: i32,

    /// The number of times this time period has been extended
    count: Arc<AtomicUsize>,

    /// The duration of this period (including any extensions)
    expire_time: Arc<AtomicUsize>,

    /// The start time of this period.
    start_time: Instant,
}

impl ByoYomi {
    pub fn new(move_number: usize, starting_visits: i32, main_time: f32, byo_yomi_time: f32, byo_yomi_periods: usize) -> ByoYomi {
        let main_time_ms = (990.0 * main_time) as usize;
        let byo_yomi_time_ms = (990.0 * byo_yomi_time) as usize;

        ByoYomi {
            total_time_ms: (main_time_ms + byo_yomi_time_ms * byo_yomi_periods).saturating_sub(*SAFE_TIME_MS),
            starting_visits: starting_visits,
            count: Arc::new(AtomicUsize::new(0)),

            start_time: Instant::now(),
            expire_time: Arc::new(AtomicUsize::new({
                let safe_main_time_ms = main_time_ms.saturating_sub(*SAFE_TIME_MS);
                let safe_byo_yomi_time_ms = if safe_main_time_ms == 0 {
                    byo_yomi_time_ms.saturating_sub(*SAFE_TIME_MS)
                } else {
                    byo_yomi_time_ms
                };

                let period_ms = if move_number < 247 {
                    // The average game length is 257 moves as suggested
                    // by _Andries E. Brouwer_:
                    //
                    // https://homepages.cwi.nl/~aeb/go/misc/gostat.html
                    let remaining_regret = regret_cost_cum(257, 257) - regret_cost_cum(move_number, 257);
                    let fraction = regret_cost(move_number, 257) / remaining_regret;

                    0.9 * fraction * safe_main_time_ms as f32
                } else {
                    // assume the game will last for 10 more moves (forever)
                    //
                    // this will decay very quickly but should be fine since
                    // past the expected end-game most moves should just be the
                    // opponent refusing to surrender and playing stupid moves.
                    0.1 * safe_main_time_ms as f32
                };

                safe_byo_yomi_time_ms + period_ms as usize
            })),
        }
    }
}

impl TimeStrategy for ByoYomi {
    fn try_extend<F: Fn() -> bool>(
        &self,
        root: &tree::Node,
        predicate: F,
        factor: f32
    ) -> TimeStrategyResult
    {
        let mut expire_time_init = self.expire_time.load(Ordering::Acquire);

        // optimistic locking using atomic values, it will have a new value for
        // `expire_time` in the beginning of each loop iteration and check if it
        // needs updating, if it does not then it exits immediately otherwise it
        // tries to update it.
        //
        // If the update fails due to concurrent modifications then it will try
        // the loop again with the updated value of `expire_time`. This should
        // be a lot cheaper than actual locking with a mutex.
        loop {
            let elapsed = self.start_time.elapsed();
            let expires = Duration::from_millis(if expire_time_init < PERIOD_BUF_TIME_MS {
                0
            } else {
                (expire_time_init - PERIOD_BUF_TIME_MS) as u64
            });

            if elapsed >= expires {
                // determine if it is possible to (and we want to) extend this
                // time period further.
                let count = self.count.load(Ordering::Acquire);
                let expire_time_next = {
                    let next = (factor * expire_time_init as f32) as usize;

                    if next > self.total_time_ms {
                        self.total_time_ms
                    } else {
                        next
                    }
                };

                if count < 2 && expire_time_next > expire_time_init && predicate() {
                    // attempt to extend this time period, checking so that no
                    // one else has done it while we worked.
                    let previous_value = self.expire_time.compare_exchange_weak(
                        expire_time_init,
                        expire_time_next,
                        Ordering::AcqRel,
                        Ordering::Relaxed
                    );

                    match previous_value {
                        Ok(_) => {
                            self.count.fetch_add(1, Ordering::Release);

                            return TimeStrategyResult::Extended;
                        },
                        Err(previous_value) => {
                            // someone else changed the value, try again!
                            expire_time_init = previous_value;
                        }
                    }
                } else {
                    return TimeStrategyResult::Expired;
                }
            } else {
                // estimate the number of remaining rollouts by checking how
                // fast they have been so far.
                //
                // Special case: If an insufficient number of rollouts has been
                // performed so far, then return `Inf` since we need more
                // samples.
                let total_visits = root.total_count - self.starting_visits;
                let elapsed_ms = (elapsed.as_secs() as f32) * 1000.0
                    + (elapsed.subsec_nanos() as f32) * 1e-6;

                return TimeStrategyResult::NotExpired(if total_visits < 5 || elapsed_ms < 1.0 {
                    ::std::usize::MAX  // unknown
                } else {
                    let rate = total_visits as f32 / elapsed_ms;
                    let remaining = expires - elapsed;
                    let remaining_ms = (remaining.as_secs() as f32) * 1000.0
                        + (remaining.subsec_nanos() as f32) * 1e-6;

                    (rate * remaining_ms) as usize
                });
            }
        }
    }
}

/// Returns the regret if the given move number ends up becoming a blunder.
///
/// # Arguments
///
/// * `move_nr` - the number of moves into the game
/// * `estimate` - the estimated number of moves this game will last
///
#[inline]
fn regret_cost(move_nr: usize, estimate: usize) -> f32 {
    let remaining = estimate - move_nr;

    debug_assert!(remaining > 0);

    2.5 * (remaining as f32 / estimate as f32)
}

/// Returns the integral of `regret_cost` from zero to `move_nr`.
///
/// # Arguments
///
/// * `move_nr` - the number of moves into the game
/// * `estimate` - the estimated number of moves this game will last
///
#[inline]
fn regret_cost_cum(move_nr: usize, estimate: usize) -> f32 {
    let move_nr = move_nr as f32;
    let estimate = estimate as f32;

    2.5 * (move_nr * estimate - 0.5 * move_nr * move_nr) / estimate
}
