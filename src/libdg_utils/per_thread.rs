// Copyright 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use dashmap::DashMap;
use dashmap::mapref::one::RefMut;
use std::sync::Arc;
use std::thread::{self, ThreadId};

#[derive(Default)]
pub struct PerThread<T: Default> {
    entries: Arc<DashMap<ThreadId, T>>
}

impl<T: Default> Clone for PerThread<T> {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone()
        }
    }
}

impl<T: Default> PerThread<T> {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(DashMap::new())
        }
    }

    pub fn get(&self) -> RefMut<'_, std::thread::ThreadId, T> {
        let thread_id = thread::current().id();
        let ref_mut = self.entries.entry(thread_id).or_default();

        ref_mut
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_twice() {
        let per_thread = PerThread::<i32>::new();

        assert_eq!(*per_thread.get(), 0);
        *per_thread.get() += 1;
        assert_eq!(*per_thread.get(), 1);

        for _ in 0..10 {
            let per_thread = per_thread.clone();

            thread::spawn(move || {
                assert_eq!(*per_thread.get(), 0);
                *per_thread.get() += 1;
                assert_eq!(*per_thread.get(), 1);
                *per_thread.get() += 2;
                assert_eq!(*per_thread.get(), 3);
            }).join().expect("could not join test thread");
        }

        assert_eq!(*per_thread.get(), 1);
    }
}
