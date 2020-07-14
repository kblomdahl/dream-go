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

use std::mem::size_of;
use std::sync::{Mutex, MutexGuard};

use dg_cuda::{PerDevice, Ptr, Stream};
use super::Error;

/// A data structure with interior mutability that store the host,
/// device, and meta information about a tensor.
pub struct Tensor {
    /// The unscaled tensor in host-memory as raw (untyped) bytes.
    host: Vec<u8>,

    /// The scaled tensor in device memory as the type given in
    /// `dtype`, or null if not applicable.
    ptr: PerDevice<Mutex<Ptr>>,

    /// The size of this tensor in bytes.
    size_in_bytes: usize,

    /// The size of this tensor in the number of elements.
    size_in_elements: usize,

    /// The scale of this tensor,
    scale: f32
}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor {
            host: vec! [],
            ptr: PerDevice::new().unwrap(),
            size_in_bytes: 0,
            size_in_elements: 0,
            scale: 1.0
        }
    }
}

impl Tensor {
    pub fn get(&self) -> MutexGuard<Ptr> {
        self.ptr.lock().unwrap()
    }

    pub fn size_in_bytes(&self) -> usize {
        self.size_in_bytes
    }

    pub fn size_in_elements(&self) -> usize {
        self.size_in_elements
    }

    #[cfg(test)]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    pub fn set_host<T: Sized>(&mut self, data: Vec<T>) -> Result<(), Error> {
        unsafe {
            let (raw_ptr, length, capacity) = data.into_raw_parts();

            self.size_in_bytes = size_of::<T>() * length;
            self.size_in_elements = length;
            self.host = Vec::from_raw_parts(raw_ptr as *mut _, self.size_in_bytes, capacity);

            Ok(())
        }
    }

    pub unsafe fn as_f32(&self) -> f32 {
        *(self.host.as_ptr() as *const f32)
    }

    pub unsafe fn as_i32(&self) -> i32 {
        *(self.host.as_ptr() as *const i32)
    }

    pub unsafe fn copy_to_device(&self, stream: &Stream) -> Result<bool, Error> {
        let mut ptr = self.ptr.lock().unwrap();

        if ptr.is_null() {
            let padded_size_in_bytes =
                if self.size_in_bytes % 32 == 0 {
                    self.size_in_bytes
                } else {
                    self.size_in_bytes + (32 - self.size_in_bytes % 32)
                };

            ptr.resize(padded_size_in_bytes)?;
            ptr.copy_from_slice(&self.host, &stream)?;

            Ok(true)
        } else {
            Ok(false)
        }
    }
}
