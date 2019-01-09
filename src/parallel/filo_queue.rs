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

use std::sync::atomic::{AtomicPtr, Ordering};
use std::mem::ManuallyDrop;
use std::ptr;

/// Lockless FILO (First In Last Out) queue based on single linked
/// lists.
pub struct FiloQueue<T> {
    /// The most recently added element to the list
    head: AtomicPtr<Head<T>>
}

struct Head<T> {
    value: ManuallyDrop<T>,
    next: AtomicPtr<Head<T>>
}

impl<T> Default for FiloQueue<T> {
    fn default() -> FiloQueue<T> {
        FiloQueue {
            head: AtomicPtr::new(ptr::null_mut())
        }
    }
}

impl<T> FiloQueue<T> {
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed).is_null()
    }

    /// Adds the given element at the beginning of the queue.
    ///
    /// # Arguments
    ///
    /// * `value` - the value to add
    ///
    pub fn push(&self, value: T) {
        let mut head = self.head.load(Ordering::Relaxed);
        let next = Box::into_raw(Box::new(Head {
            value: ManuallyDrop::new(value),
            next: AtomicPtr::new(head)
        }));

        loop {
            let other = self.head.compare_and_swap(head, next, Ordering::SeqCst);

            if head == other {
                break
            }

            unsafe {
                head = other;
                (*next).next = AtomicPtr::new(other);
            }
        }
    }

    /// Removes and returns the most recently added element to the queue.
    pub fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Relaxed);
            if head.is_null() {
                return None;
            }

            let next = unsafe {  (*head).next.load(Ordering::Relaxed) };
            let other = self.head.compare_and_swap(head, next, Ordering::SeqCst);

            if other == head {
                let mut head_ref = unsafe { Box::from_raw(head) };
                let value = unsafe { ManuallyDrop::take(&mut head_ref.value) };

                return Some(value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty() {
        let x: FiloQueue<i8> = FiloQueue::default();

        assert!(x.is_empty());
    }

    #[test]
    fn push_pop_is_empty() {
        let x = FiloQueue::default();

        for i in 0..10 {
            x.push(i);
        }

        for i in 0..10 {
            assert_eq!(x.pop(), Some(9 - i));
        }
        assert!(x.is_empty());
    }
}