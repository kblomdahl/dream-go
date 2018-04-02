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

use mcts::time_control::{TimeStrategy, TimeStrategyResult};
use mcts::tree;

#[derive(Clone)]
pub struct RolloutLimit {
    limit: i32
}

impl RolloutLimit {
    pub fn new(limit: usize) -> RolloutLimit {
        RolloutLimit {
            limit: if limit > ::std::i32::MAX as usize {
                ::std::i32::MAX - 1
            } else {
                limit as i32
            }
        }
    }
}

impl TimeStrategy for RolloutLimit {
    fn try_extend<E: tree::Value, F: Fn() -> bool>(
        &self,
        root: &tree::Node<E>,
        _predicate: F,
        _factor: f32
    ) -> TimeStrategyResult
    {
        if root.total_count < self.limit {
            let remaining = self.limit - root.total_count;

            TimeStrategyResult::NotExpired(remaining as usize)
        } else {
            TimeStrategyResult::Expired
        }
    }
}
