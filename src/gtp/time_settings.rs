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

pub trait TimeSettings {
    /// Update the amount of remaining time according to the canadian
    /// rules, where the `main_time` indicates the amount of time remaining and
    /// `byo_yomi_stones` is the number of stones left until another period
    /// is granted.
    /// 
    /// # Arguments
    /// 
    /// * `main_time` -
    /// * `byo_yomi_stones` -
    /// 
    fn time_left(&mut self, main_time: f32, byo_yomi_stones: usize);

    /// Returns the amount of remaining time as the
    /// triplet `(main_time, byo_yomi_time, byo_yomi_periods)` according to
    /// the byo yomi time system.
    fn remaining(&self) -> (f32, f32);

    /// Decrease the amount of remaining time by the given amount of seconds.
    /// 
    /// # Argument
    /// 
    /// * `elapsed` -
    /// 
    fn update(&mut self, elapsed: f32);
}

// -------- Infinite Thinking Time --------

/// An implementation of `TimeSettings` that allows for infinite thinking
/// time. If you use this one then you probably want to add some other
/// constraint such as a limit on the number of rollouts as otherwise
/// it **will** think forever.
pub struct None;

impl None {
    pub fn new() -> None {
        None {}
    }
}

impl TimeSettings for None {
    fn time_left(&mut self, _main_time: f32, _byo_yomi_stones: usize) {
        // pass
    }

    fn remaining(&self) -> (f32, f32) {
        (::std::f32::INFINITY, ::std::f32::INFINITY)
    }

    fn update(&mut self, _elapsed: f32) {
        // pass
    }
}

// -------- Sudden Death Time --------

/// An implementation of `TimeSettings` which has only one period, and whoever
/// runs out of time loses. There is no way to gain additional time.
pub struct Absolute {
    main_time: f32
}

impl Absolute {
    pub fn new(main_time: f32) -> Absolute {
        Absolute {
            main_time: main_time
        }
    }
}

impl TimeSettings for Absolute {
    fn time_left(&mut self, main_time: f32, _byo_yomi_stones: usize) {
        self.main_time = main_time;
    }

    fn remaining(&self) -> (f32, f32) {
        (self.main_time, 0.0)
    }

    fn update(&mut self, elapsed: f32) {
        self.main_time -= elapsed;
        if self.main_time < 0.0 {
            self.main_time = 0.0;
        }
    }
}

// -------- Byo Yomi Time --------

/// An implementation of `TimeSettings` that uses the byo-yomi time system where
/// one has a fixed amount of time per move.
pub struct ByoYomi {
    main_time: f32,
    byo_yomi_time: f32,
    byo_yomi_periods: usize
}

impl ByoYomi {
    pub fn new(main_time: f32, byo_yomi_time: f32, byo_yomi_periods: usize) -> ByoYomi {
        ByoYomi {
            main_time: main_time,
            byo_yomi_time: byo_yomi_time,
            byo_yomi_periods: byo_yomi_periods
        }
    }
}

impl TimeSettings for ByoYomi {
    fn time_left(&mut self, main_time: f32, byo_yomi_stones: usize) {
        if byo_yomi_stones == 0 {
            self.main_time = main_time;
        } else {
            self.byo_yomi_time = main_time;
        }
    }

    fn remaining(&self) -> (f32, f32) {
        (self.main_time, self.byo_yomi_time * self.byo_yomi_periods as f32)
    }

    fn update(&mut self, elapsed: f32) {
        self.main_time -= elapsed;
        if self.main_time < 0.0 {
            let mut overtime = -self.main_time;

            while overtime > self.byo_yomi_time {
                overtime -= self.byo_yomi_time;
                self.byo_yomi_periods -= 1;
            }

            self.main_time = 0.0;
        }
    }
}


// -------- Canadian Time --------

/// An implementation of `TimeSettings` that uses Canadian time, where one
/// starts with some amount of time and then gains extra time for
/// every _n_ moves.
pub struct Canadian  {
    main_time: f32,
    byo_yomi_time: f32,
    byo_yomi_stones: usize,
    byo_yomi_stones_remaining: usize
}

impl Canadian {
    pub fn new(main_time: f32, byo_yomi_time: f32, byo_yomi_stones: usize) -> Canadian {
        Canadian {
            main_time: main_time,
            byo_yomi_time: byo_yomi_time,
            byo_yomi_stones: byo_yomi_stones,
            byo_yomi_stones_remaining: byo_yomi_stones
        }
    }
}

impl TimeSettings for Canadian {
    fn time_left(&mut self, main_time: f32, byo_yomi_stones: usize) {
        self.main_time = main_time;
        self.byo_yomi_stones_remaining = byo_yomi_stones;
    }

    fn remaining(&self) -> (f32, f32) {
        (self.main_time, 0.0)
    }

    fn update(&mut self, elapsed: f32) {
        self.main_time -= elapsed;
        if self.main_time < 0.0 {
            self.main_time = 0.0;
        }

        // increase the remaining time by the `byo_yomi_time` if we've played
        // the required number of stones
        self.byo_yomi_stones_remaining -= 1;

        if self.byo_yomi_stones_remaining == 0 {
            self.byo_yomi_stones_remaining = self.byo_yomi_stones;
            self.main_time += self.byo_yomi_time;
        }
    }
}
