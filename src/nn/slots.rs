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

    /// Returns a pointer to a named additional variable. If two variables
    /// shares the same name, then their pointers may alias.
    /// 
    /// # Arguments
    /// 
    /// * `name` - the name of the variable
    /// * `size_in_bytes` - the minimum required size of the allocated area
    pub fn get_slot(&self, name: Slot, size_in_bytes: usize) -> SlotGuard {
        let device_id = get_current_device();
        let mut inner = self.inner.lock().unwrap();
        let slot = {
            let device_id = device_id as usize;
            let slot_pos = name as usize;

            while inner[device_id].len() <= slot_pos {
                inner[device_id].push(SlotInner {
                    ptr: vec! [],
                    size_in_bytes: 0
                });
            }

            &mut inner[device_id][slot_pos]
        };

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
            debug_assert!(slot.size_in_bytes == 0 || !ptr.is_null());

            SlotGuard {
                name: name,
                device_id: device_id,
                ptr: Rc::new(ptr),
                size_in_bytes: slot.size_in_bytes,
                inner: Some(self.inner.clone())
            }
        } else {
            let mut ptr: *mut c_void = ptr::null_mut();

            unsafe {
                check!(cuda::cudaMalloc(&mut ptr, slot.size_in_bytes));
                check!(cuda::cudaMemset(ptr, 0, slot.size_in_bytes));

                debug_assert!(slot.size_in_bytes == 0 || !ptr.is_null(), "Failed to allocate CUDA buffer of size {} (expected size {})", slot.size_in_bytes, size_in_bytes);
            }

            SlotGuard {
                name: name,
                device_id: device_id,
                ptr: Rc::new(ptr),
                size_in_bytes: slot.size_in_bytes,
                inner: Some(self.inner.clone())
            }
        }
    }
}

#[derive(Clone)]
pub struct SlotGuard {
    name: Slot,
    device_id: i32,
    ptr: Rc<*mut c_void>,
    size_in_bytes: usize,

    inner: Option<Arc<Mutex<SlotsInner>>>
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if Rc::strong_count(&self.ptr) == 1 {
            if let Some(ref inner) = self.inner {
                let mut inner = inner.lock().unwrap();
                let slot = &mut inner[self.device_id as usize][self.name as usize];

                if self.size_in_bytes < slot.size_in_bytes {
                    // if the slot has grown in our absence then throw our pointer away
                    unsafe {
                        cuda::cudaFree(*self.ptr);
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
