// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use libc::c_void;
use std::ptr;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use nn::ffi::cuda;

struct Slot {
    ptr: Vec<*mut c_void>,
    size_in_bytes: usize,
}

struct SlotsInner {
    named: HashMap<String, Slot>
}

impl Drop for SlotsInner {
    fn drop(&mut self) {
        for slot in self.named.values() {
            for ptr in &slot.ptr {
                unsafe {
                    check!(cuda::cudaFree(*ptr));
                }
            }
        }
    }
}

/// CUDA memory pool that keep requires all objects in the same pool to be of
/// the same minimum size.
#[derive(Clone)]
pub struct Slots {
    inner: Arc<Mutex<SlotsInner>>
}

impl Slots {
    pub fn new() -> Slots {
        Slots {
            inner: Arc::new(Mutex::new(SlotsInner {
                named: HashMap::new()
            }))
        }
    }

    /// Returns a pointer to a named additional variable. If two variables
    /// shares the same name, then their pointers may alias.
    /// 
    /// # Arguments
    /// 
    /// * `name` - the name of the variable
    /// * `size_in_bytes` - the minimum required size of the allocated area
    pub fn get_slot(&self, name: &'static str, size_in_bytes: usize) -> SlotGuard {
        let mut inner = self.inner.lock().unwrap();
        let slot = inner.named.entry(name.to_string()).or_insert_with(|| {
            Slot {
                ptr: vec! [],
                size_in_bytes: size_in_bytes
            }
        });

        if slot.size_in_bytes < size_in_bytes {
            // all of the stored pointers are too small, they will need to be
            // re-allocated. But that will be done on-demand, here we just free
            // them
            let num_ptr = slot.ptr.len();

            for p in slot.ptr.drain(0..num_ptr) {
                unsafe {
                    check!(cuda::cudaFree(p));
                }
            }

            slot.size_in_bytes = size_in_bytes;
        }

        if let Some(ptr) = slot.ptr.pop() {
            SlotGuard {
                name: name.to_string(),
                ptr: Rc::new(ptr),
                size_in_bytes: slot.size_in_bytes,
                inner: Some(self.inner.clone())
            }
        } else {
            let mut ptr: *mut c_void = ptr::null_mut();

            unsafe {
                check!(cuda::cudaMalloc(&mut ptr, slot.size_in_bytes));
            }

            SlotGuard {
                name: name.to_string(),
                ptr: Rc::new(ptr),
                size_in_bytes: slot.size_in_bytes,
                inner: Some(self.inner.clone())
            }
        }
    }
}

#[derive(Clone)]
pub struct SlotGuard {
    name: String,
    ptr: Rc<*mut c_void>,
    size_in_bytes: usize,

    inner: Option<Arc<Mutex<SlotsInner>>>
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if Rc::strong_count(&self.ptr) == 1 {
            if let Some(ref inner) = self.inner {
                let mut inner = inner.lock().unwrap();
                let slot = inner.named.get_mut(&self.name).unwrap();

                if self.size_in_bytes < slot.size_in_bytes {
                    // if the slot has grown in our absence then throw our pointer away
                    unsafe {
                        check!(cuda::cudaFree(*self.ptr));
                    }
                } else if self.size_in_bytes == slot.size_in_bytes {
                    // return this pointer to the pool
                    slot.ptr.push(*self.ptr);
                } else {
                    unreachable!();
                }
            } else {
                debug_assert!(self.ptr.is_null());
            }
        }
    }
}

impl<'a> ::std::ops::Deref for SlotGuard {
    type Target = *mut c_void;

    fn deref(&self) -> &*mut c_void {
        &self.ptr
    }
}

impl SlotGuard {
    pub fn null() -> SlotGuard {
        SlotGuard {
            name: "null".to_string(),
            ptr: Rc::new(ptr::null_mut()),
            size_in_bytes: 0,

            inner: None
        }
    }
}