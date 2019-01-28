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

use std::sync::{RwLock, RwLockReadGuard};
use std::cell::RefCell;

/// The state of the current thread, whether it is holding the read-lock or not.
enum ThreadLockState {
    None,
    Read(RwLockReadGuard<'static, ()>)
}

lazy_static! {
    /// The global rw-lock
    static ref RWLOCK: RwLock<()> = RwLock::new(());
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
/// * `f` - the function to execute inside of a _write lock_.
///
pub fn write<F: FnOnce()>(f: F) -> bool {
    let was_reader = read_unlock();

    {
        let _guard = RWLOCK.write().unwrap();

        f();
    }

    if was_reader {
        read_lock()
    } else {
        false
    }
}

/// Acquire a _read lock_ for the current thread. Returns `true` iff this thread did not already
/// hold a _read lock_.
pub fn read_lock() -> bool {
    LOCK_STATE.with(|state| {
        let next_state = match *state.borrow() {
            ThreadLockState::Read(ref _guard) => { return false },
            ThreadLockState::None => {
                ThreadLockState::Read(RWLOCK.read().unwrap())
            }
        };

        *state.borrow_mut() = next_state;
        true
    })
}

/// Release the _read lock_ for the current thread. Returns `true` iff this thread held the
/// _read lock_.
pub fn read_unlock() -> bool {
    LOCK_STATE.with(|state| {
        let next_state = match *state.borrow() {
            ThreadLockState::None => { return false },
            ThreadLockState::Read(ref _guard) => {
                ThreadLockState::None
            }
        };

        *state.borrow_mut() = next_state;
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promote_read() {
        assert_eq!(read_lock(), true);
        assert_eq!(write(|| { }), true);
        assert_eq!(read_unlock(), true);
    }

    #[test]
    fn write_without_read() {
        assert_eq!(write(|| { }), false);
    }

    #[test]
    fn register_already_registered() {
        assert_eq!(read_lock(), true);
        assert_eq!(read_lock(), false);
    }

    #[test]
    fn unregister_not_registered() {
        assert_eq!(read_unlock(), false);
    }
}
