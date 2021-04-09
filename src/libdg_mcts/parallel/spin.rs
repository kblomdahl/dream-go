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

use crossbeam_utils::Backoff;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct MutexGuard<'a> {
    is_available: &'a AtomicBool
}

impl<'a> Drop for MutexGuard<'a> {
    fn drop(&mut self) {
        self.is_available.store(true, Ordering::Release);
    }
}

/// A lock that provides mutual access using a spinlock algorithm, this makes
/// it suitable for locks that will only be held for *very brief* periods of
/// time.
pub struct Mutex {
    is_available: AtomicBool
}

impl Mutex {
    /// Returns an unlocked mutex.
    pub fn new() -> Mutex {
        Mutex { is_available: AtomicBool::new(true) }
    }

    #[inline]
    pub fn lock(&self) -> MutexGuard {
        let backoff = Backoff::new();

        loop {
            match self.is_available.compare_exchange_weak(true, false, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => break,
                _ => { backoff.snooze(); }
            }
        }

        MutexGuard { is_available: &self.is_available }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock() {
        let mutex = Mutex::new();

        mutex.lock();
    }
}
