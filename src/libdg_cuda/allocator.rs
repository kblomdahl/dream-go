// Copyright 2020 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use crate::error::Error;
use crate::memory::*;

use std::cell::{RefCell, Ref};
use std::collections::btree_map::BTreeMap;
use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::ops::{Deref, DerefMut};

pub trait Allocator {
    fn alloc(&mut self, size_in_bytes: usize) -> Result<Ptr, Error>;
    fn free(&mut self, ptr: Ptr);
}

#[derive(Clone, Default)]
pub struct Native {
    // pass
}

impl Native {
    pub fn new() -> Self {
        Self {}
    }
}

impl Allocator for Native {
    fn alloc(&mut self, size_in_bytes: usize) -> Result<Ptr, Error> {
        Ptr::new(size_in_bytes)
    }

    fn free(&mut self, ptr: Ptr) {
        drop(ptr);
    }
}

pub struct Sticky<A: Allocator> {
    allocator: A,
    ptrs: BTreeMap<usize, Vec<Ptr>>
}

impl<A: Allocator + Default> Default for Sticky<A> {
    fn default() -> Self {
        Self::new(A::default())
    }
}

impl<A: Allocator> Sticky<A> {
    pub fn new(allocator: A) -> Self {
        let ptrs = BTreeMap::new();
        Self { allocator, ptrs }
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }
}

impl<A: Allocator> Drop for Sticky<A> {
    fn drop(&mut self) {
        while let Some(&x) = self.ptrs.keys().next() {
            for ptr in self.ptrs.remove(&x).unwrap().drain(..) {
                self.allocator.free(ptr);
            }
        }
    }
}

impl<A: Allocator> Allocator for Sticky<A> {
    fn alloc(&mut self, size_in_bytes: usize) -> Result<Ptr, Error> {
        match self.ptrs.get_mut(&size_in_bytes) {
            Some(x) if x.len() > 0 => Ok(x.pop().unwrap()),
            _ => self.allocator.alloc(size_in_bytes)
        }
    }

    fn free(&mut self, ptr: Ptr) {
        self.ptrs.entry(ptr.size_in_bytes()).or_default().push(ptr);
    }
}

pub struct Concurrent<A: Allocator> {
    allocator: Arc<Mutex<A>>
}

impl<A: Allocator> Clone for Concurrent<A> {
    fn clone(&self) -> Self {
        Self { allocator: Arc::clone(&self.allocator) }
    }
}

impl<A: Allocator> Concurrent<A> {
    pub fn new(allocator: A) -> Self {
        Self { allocator: Arc::new(Mutex::new(allocator)) }
    }
}

impl<A: Allocator + Default> Default for Concurrent<A> {
    fn default() -> Self {
        Self::new(A::default())
    }
}

impl<A: Allocator> Allocator for Concurrent<A> {
    fn alloc(&mut self, size_in_bytes: usize) -> Result<Ptr, Error> {
        self.allocator.lock().unwrap().alloc(size_in_bytes)
    }

    fn free(&mut self, ptr: Ptr) {
        self.allocator.lock().unwrap().free(ptr)
    }
}

pub struct Cloneable<A: Allocator> {
    allocator: Rc<RefCell<A>>
}

impl<A: Allocator + Default> Default for Cloneable<A> {
    fn default() -> Self {
        Self::new(A::default())
    }
}

impl<A: Allocator> Clone for Cloneable<A> {
    fn clone(&self) -> Self {
        let allocator = Rc::clone(&self.allocator);

        Self { allocator }
    }
}

impl<A: Allocator> Cloneable<A> {
    pub fn new(allocator: A) -> Self {
        Self { allocator: Rc::new(RefCell::new(allocator)) }
    }

    pub fn allocator(&self) -> Ref<'_, A> {
        self.allocator.borrow()
    }
}

impl<A: Allocator> Allocator for Cloneable<A> {
    fn alloc(&mut self, size_in_bytes: usize) -> Result<Ptr, Error> {
        self.allocator.borrow_mut().alloc(size_in_bytes)
    }

    fn free(&mut self, ptr: Ptr) {
        self.allocator.borrow_mut().free(ptr)
    }
}

pub struct SmartPtr<A: Allocator> {
    allocator: A,
    ptr: Option<Ptr>
}

impl<'a, A: Allocator> Deref for SmartPtr<A> {
    type Target = Ptr;

    fn deref(&self) -> &Self::Target {
        &self.ptr.as_ref().unwrap()
    }
}

impl<'a, A: Allocator> DerefMut for SmartPtr<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ptr.as_mut().unwrap()
    }
}

impl<'a, A: Allocator> Drop for SmartPtr<A> {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr.take() {
            self.allocator.free(ptr);
        }
    }
}

impl<A: Allocator> SmartPtr<A> {
    pub fn unwrap(mut self) -> Ptr {
        self.ptr.take().unwrap()
    }
}

pub fn malloc<'a, A: Allocator + Clone>(
    size_in_bytes: usize,
    allocator: &A
) -> Result<SmartPtr<A>, Error>
{
    let mut allocator = allocator.clone();
    let ptr = Some(allocator.alloc(size_in_bytes)?);

    Ok(SmartPtr { allocator, ptr })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingAllocator {
        pub total_allocs: usize,
        pub total_frees: usize
    }

    impl Allocator for CountingAllocator {
        fn alloc(&mut self, size_in_bytes: usize) -> Result<Ptr, Error> {
            self.total_allocs += 1;
            Ptr::new(size_in_bytes)
        }

        fn free(&mut self, _ptr: Ptr) {
            self.total_frees += 1;
        }
    }

    #[test]
    fn native() {
        let mut allocator = Native::new();
        let ptr = allocator.alloc(12).unwrap();
        assert_eq!(ptr.size_in_bytes(), 12);
        allocator.free(ptr);
    }

    #[test]
    fn sticky() {
        let counting_allocator = CountingAllocator { total_allocs: 0, total_frees: 0 };
        let mut allocator = Sticky::new(counting_allocator);

        for _i in 0..10 {
            let ptr = allocator.alloc(12).unwrap();
            assert_eq!(ptr.size_in_bytes(), 12);
            allocator.free(ptr);
        }

        assert_eq!(allocator.allocator().total_allocs, 1);
        assert_eq!(allocator.allocator().total_frees, 0);
    }

    #[test]
    fn smart_ptr() {
        let counting_allocator = CountingAllocator { total_allocs: 0, total_frees: 0 };
        let allocator = Cloneable::new(counting_allocator);
        let smart_ptr = malloc(12, &allocator).unwrap();

        assert_eq!(allocator.allocator().total_allocs, 1);
        assert_eq!(allocator.allocator().total_frees, 0);
        drop(smart_ptr);
        assert_eq!(allocator.allocator().total_frees, 1);
    }
}
