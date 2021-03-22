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

use std::collections::HashMap;
use std::sync::Mutex;
use std::hash::{Hash, Hasher};
use std::ptr;

use dg_go::utils::symmetry;
use dg_go::{Board, Color};

/// The maximum number of entries to be stored in the transposition table
/// before we need to remove the least recently used one.
const MAX_CACHE_SIZE: usize = 200_000;

#[derive(Debug)]
struct KeyRef<K: Hash + Eq> {
    inner: *const K
}

impl<K: Hash + Eq> Hash for KeyRef<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe { (*self.inner).hash(state) };
    }
}

impl<K: Hash + Eq> PartialEq for KeyRef<K> {
    fn eq(&self, other: &KeyRef<K>) -> bool {
        unsafe { (*self.inner).eq(&*(other.inner)) }
    }
}

impl<K: Hash + Eq> Eq for KeyRef<K> { }

#[derive(Debug)]
struct LruEntry<K: Clone, V: Clone> {
    key: K,
    value: V,

    next: *mut LruEntry<K, V>,
    prev: *mut LruEntry<K, V>
}

/// Wrapper around `HashMap` that only keeps the most recently
/// added or accessed elements.
///
/// This is implemented as a double linked hashmap, where new elements
/// are added or moves to the head when accessed and if the list becomes
/// too large then the tail of the list is removed from the map.
#[derive(Debug)]
struct LruCache<K: Clone + Hash + Eq, V: Clone> {
    entries: HashMap<KeyRef<K>, Box<LruEntry<K, V>>>,
    capacity: usize,
    head: *mut LruEntry<K, V>,
    tail: *mut LruEntry<K, V>
}

unsafe impl<K: Clone + Hash + Eq, V: Clone> Send for LruCache<K, V> { }

/// Add the given entry as the most recently accessed key.
macro_rules! attach {
    ($self:expr, $entry:expr) => ({
        $entry.prev = ptr::null_mut();
        $entry.next = $self.head;

        if !$self.head.is_null() {
            (*$self.head).prev = $entry;
        }

        $self.head = $entry;
        if $self.tail.is_null() {
            $self.tail = $entry;
        }
    });
}

/// Removes the given entry from the list of most recently accessed
/// keys.
macro_rules! detach {
    ($self:expr, $entry:expr) => ({
        if !$entry.prev.is_null() { (*$entry.prev).next = $entry.next; }
        if !$entry.next.is_null() { (*$entry.next).prev = $entry.prev; }

        if $self.head == $entry { $self.head = $entry.next; }
        if $self.tail == $entry { $self.tail = $entry.prev; }
    });
}

impl<K: Clone + Hash + Eq, V: Clone> LruCache<K, V> {
    fn with_capacity(cap: usize) -> LruCache<K, V> {
        LruCache {
            entries: HashMap::with_capacity(cap),
            capacity: cap,
            head: ptr::null_mut(),
            tail: ptr::null_mut()
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        let key_ref = KeyRef { inner: key };

        if let Some(entry) = self.entries.get_mut(&key_ref) {
            unsafe {
                detach!(self, &mut **entry);
                attach!(self, &mut **entry);
            }

            Some(&entry.value)
        } else {
            None
        }
    }

    fn insert(&mut self, key: &K, value: V) {
        let key_ref = KeyRef { inner: key };

        if !self.entries.contains_key(&key_ref) {
            let mut entry = Box::new(LruEntry {
                key: key.clone(),
                value: value,

                prev: ptr::null_mut(),
                next: self.head
            });

            unsafe { attach!(self, &mut *entry) };
            self.entries.insert(KeyRef { inner: &entry.key }, entry);

            // if the cache is too large then drop the tail in order to
            // preserve the max size
            if self.entries.len() > self.capacity {
                unsafe {
                    let tail = &mut *self.tail;

                    detach!(self, tail);
                    self.entries.remove(&KeyRef { inner: &tail.key });
                }
            }
        }
    }
}

/* -------- get_or_insert -------- */

#[derive(Clone, Hash, PartialEq, Eq)]
struct BoardTuple {
    board: Board,
    to_move: Color,
    symmetry: symmetry::Transform
}

/// Retrieve the value and policy from the transposition table, if
/// the `(board, color)`  tuple is not in the transposition table then
/// it is computed from the given supplier.
///
/// # Arguments
///
/// * `board` - the board to get from the table
/// * `to_move` - the color to get from the table
/// * `symmetry` - the symmetry to get from the table
/// * `supplier` - a function that can be used to compute the value
///   and policy if they are missing from the table.
///
pub fn get_or_insert<F>(
    board: &Board,
    to_move: Color,
    symmetry: symmetry::Transform,
    supplier: F
) -> Option<(f32, Vec<f32>)>
    where F: FnOnce() -> Option<(f32, Vec<f32>)>
{
    lazy_static! {
        static ref TABLE: Mutex<LruCache<BoardTuple, (f32, Vec<f32>)>> = {
            Mutex::new(LruCache::with_capacity(MAX_CACHE_SIZE + 1))
        };
    }

    let key = BoardTuple {
        board: board.clone(),
        to_move: to_move,
        symmetry: symmetry
    };
    let existing = {
        let mut table = TABLE.lock().unwrap();

        table.get(&key).map(|&(value, ref policy)| {
            (value, policy.clone())
        })
    };

    if let Some((value, policy)) = existing {
        Some((value, policy))
    } else if let Some((value, policy)) = supplier() {
        let mut table = TABLE.lock().unwrap();

        table.insert(&key, (value, policy.clone()));

        Some((value, policy))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_only_20000() {
        let mut lru = LruCache::with_capacity(1000);

        for i in 0..20000 {
            lru.insert(&i, i);
        }

        assert_eq!(lru.len(), 1000);
    }

    #[test]
    fn mixed_insert() {
        let mut lru = LruCache::with_capacity(10);

        for i in 0..10 { lru.insert(&i, i); }
        for i in 0..2 { lru.get(&i); }
        for i in 0..6 { lru.insert(&(i + 20), i); }

        assert!(lru.get(&0).is_some(), "{:?}", lru);
        assert!(lru.get(&1).is_some(), "{:?}", lru);
        assert!(lru.get(&8).is_some(), "{:?}", lru);
        assert!(lru.get(&9).is_some(), "{:?}", lru);

        for i in 2..8 {
            assert!(lru.get(&i).is_none(), "{:?}", lru);
        }
    }
}
