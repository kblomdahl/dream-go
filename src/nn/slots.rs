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
use std::cell::RefCell;
use std::ptr;
use std::sync::{Arc, Mutex};

use nn::devices::{MAX_DEVICES, get_current_device};
use nn::ffi::cuda;

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub enum Slot {
    Input = 0,
    Policy_1 = 1,
    Policy_2 = 2,
    Policy_3 = 3,
    Value_1 = 4,
    Value_2 = 5,
    Value_3 = 6,
    Residual_1 = 7,
    Residual_2 = 8,
    Workspace_1 = 9,  // workspace for tower head
    Workspace_r = 10,  // workspace for residual blocks
    Workspace_p = 11,  // workspace for policy head
    Workspace_v = 12,  // workspace for value head
}

struct SlotInner {
    ptr: Vec<*mut c_void>,
    size_in_bytes: usize,
}

type SlotsInner = [Vec<SlotInner>; MAX_DEVICES];

/// CUDA memory pool that keep requires all objects in the same pool to be of
/// the same minimum size.
#[derive(Clone)]
pub struct Slots {
    inner: Arc<Mutex<SlotsInner>>
}

impl Slots {
    pub fn new() -> Slots {
        Slots {
            inner: Arc::new(Mutex::new([
                vec! [], vec! [], vec! [], vec! [],
                vec! [], vec! [], vec! [], vec! [],
            ]))
        }
    }

    pub fn lock(&self) -> SlotsGuard {
        let device_id = get_current_device() as usize;

        SlotsGuard {
            device_id: device_id,

            pool: self.inner.clone(),
            inner: RefCell::new(vec! [])
        }
    }
}

pub struct SlotsGuard {
    device_id: usize,

    pool: Arc<Mutex<SlotsInner>>,
    inner: RefCell<Vec<SlotInner>>
}

impl Drop for SlotsGuard {
    fn drop(&mut self) {
        let global = &mut self.pool.lock().unwrap()[self.device_id];
        let mut inner = self.inner.borrow_mut();
        let used_slots = inner.iter_mut().enumerate();

        for (i, slot) in used_slots.filter(|(_, slot)| slot.size_in_bytes > 0) {
            while global.len() <= i {
                global.push(SlotInner {
                    ptr: vec! [],
                    size_in_bytes: 0
                });
            }

            let global_slot = &mut global[i];

            if global_slot.size_in_bytes < slot.size_in_bytes {
                // if all of the refs in the global pool are too small, then
                // throw them all away and replace them with out refs
                for ptr in global_slot.ptr.splice(.., slot.ptr.drain(..)) {
                    unsafe { check!(cuda::cudaFree(ptr)) };
                }

                global_slot.size_in_bytes = slot.size_in_bytes;
            } else if slot.size_in_bytes < global_slot.size_in_bytes {
                // the global pool grew in our absence, so throw away our
                // refs
                for ptr in slot.ptr.drain(..) {
                    unsafe { check!(cuda::cudaFree(ptr)) };
                }
            } else {
                debug_assert!(slot.size_in_bytes == global_slot.size_in_bytes);

                global_slot.ptr.extend(slot.ptr.drain(..));
            }
        }
    }
}

impl SlotsGuard {
    /// Returns a pointer to a named additional variable. If two variables
    /// shares the same name, then their pointers may alias.
    /// 
    /// # Arguments
    /// 
    /// * `name` - the name of the variable
    /// * `size_in_bytes` - the minimum required size of the allocated area
    /// * `stream` - the stream that will use the memory
    /// 
    pub fn get_slot<'a>(&'a self, name: Slot, size_in_bytes: usize, stream: cuda::Stream) -> SlotGuard<'a> {
        let slot_pos = name as usize;

        // check if the slot exists in the local pool
        let mut inner = self.inner.borrow_mut();

        if inner.len() > slot_pos && !inner[slot_pos].ptr.is_empty() {
            let ref mut slot = inner[slot_pos];
            let ptr = slot.ptr.pop().unwrap();

            return SlotGuard {
                name: name,
                ptr: ptr,
                size_in_bytes: slot.size_in_bytes,
                inner: &self.inner
            };
        }

        // if there is no local slot, then move (or allocate) one slot from the
        // global pool to the local pool
        let mut global = self.pool.lock().unwrap();
        let slot = {
            let device_id = self.device_id as usize;
            let global = &mut global[device_id];

            if global.len() <= slot_pos {
                None
            } else {
                let slot = &mut global[slot_pos];

                if slot.size_in_bytes < size_in_bytes || slot.ptr.is_empty() {
                    None
                } else {
                    Some(slot)
                }
            }
        };

        if let Some(slot) = slot {
            let ptr = slot.ptr.pop().unwrap();

            debug_assert!(slot.size_in_bytes == 0 || !ptr.is_null());

            SlotGuard {
                name: name,
                ptr: ptr,
                size_in_bytes: slot.size_in_bytes,
                inner: &self.inner
            }
        } else {
            let mut ptr: *mut c_void = ptr::null_mut();

            unsafe {
                check!(cuda::cudaMalloc(&mut ptr, size_in_bytes));
                check!(cuda::cudaMemsetAsync(ptr, 0, size_in_bytes, stream));

                debug_assert!(size_in_bytes == 0 || !ptr.is_null(), "Failed to allocate CUDA buffer of size {}", size_in_bytes);
            }

            SlotGuard {
                name: name,
                ptr: ptr,
                size_in_bytes: size_in_bytes,
                inner: &self.inner
            }
        }
    }
}

pub struct SlotGuard<'a> {
    name: Slot,
    ptr: *mut c_void,
    size_in_bytes: usize,

    inner: &'a RefCell<Vec<SlotInner>>
}

impl<'a> Drop for SlotGuard<'a> {
    fn drop(&mut self) {
        let mut inner = self.inner.borrow_mut();
        let slot = {
            let slot_pos = self.name as usize;

            while inner.len() <= slot_pos {
                inner.push(SlotInner {
                    ptr: vec! [],
                    size_in_bytes: 0
                })
            }

            &mut inner[slot_pos]
        };

        if self.size_in_bytes < slot.size_in_bytes {
            unreachable!();
        } else if self.size_in_bytes > slot.size_in_bytes {
            debug_assert!(slot.ptr.is_empty());

            slot.ptr.push(self.ptr);
            slot.size_in_bytes = self.size_in_bytes;
        } else {
            debug_assert!(self.size_in_bytes == slot.size_in_bytes);

            slot.ptr.push(self.ptr);
        }
    }
}

impl<'a> ::std::ops::Deref for SlotGuard<'a> {
    type Target = *mut c_void;

    fn deref(&self) -> &*mut c_void {
        &self.ptr
    }
}