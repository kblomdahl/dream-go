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

use crossbeam_utils::sync::{ShardedLock, ShardedLockReadGuard};
use std::cell::RefCell;

/// The state of the current thread, whether it is holding the read-lock or not.
type ThreadLockState = Option<ShardedLockReadGuard<'static, ()>>;

lazy_static! {
    /// The global rw-lock
    static ref RWLOCK: ShardedLock<()> = ShardedLock::new(());
}

thread_local! {
    static LOCK_STATE: RefCell<ThreadLockState> = RefCell::new(ThreadLockState::None);
}

/// Execute the given function inside of a _write-lock_. If this thread holds a _read lock_, then
/// that lock is temporarily released and then reacquired after the function has been executed.
///
/// If a _read lock_ was acquired at the end of this function it returns `true`.
///
/// # Arguments
///
/// * `callback` - the function to execute inside of a _write lock_.
///
pub fn write<F: FnOnce()>(callback: F) {
    LOCK_STATE.with(|state| {
        let next_state = match state.borrow_mut().take() {
            None => {
                let _guard = RWLOCK.write();
                return callback();
            },
            Some(guard) => {
                drop(guard);
                let guard = RWLOCK.write().expect("could not acquire write lock");
                callback();
                drop(guard);

                Some(RWLOCK.read().expect("could not acquire read lock"))
            }
        };

        *state.borrow_mut() = next_state;
    });
}

/// Acquire and hold the read-lock for the duration of the execution of the given callback, once
/// the callback has been executed release the read-lock.
///
/// # Arguments
///
/// * `callback` -
///
pub fn read<T, F: FnOnce() -> T>(callback: F) -> T {
    LOCK_STATE.with(|state| {
        let next_state = match *state.borrow() {
            Some(ref _guard) => {
                return callback();
            },
            None => {
                Some(RWLOCK.read().expect("could not acquire read lock"))
            }
        };

        *state.borrow_mut() = next_state;
        let result = callback();
        *state.borrow_mut() = None;
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_without_read() {
        write(|| { });
    }

    #[test]
    fn read_without_write() {
        assert_eq!(read(|| { false }), false);
    }

    #[test]
    fn read_with_write() {
        assert_eq!(read(|| { write(|| {}) }), ());
    }
}
