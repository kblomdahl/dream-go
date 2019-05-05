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
use std::mem::size_of;
use std::ptr;
use libc::c_void;

use super::devices::MAX_DEVICES;
use super::ffi::cuda;
use super::Error;

/// A data structure with interior mutability that store the host,
/// device, and meta information about a tensor.
pub struct Tensor {
    /// The unscaled tensor in host-memory as raw (untyped) bytes.
    pub host: *mut c_void,

    /// The scaled tensor in device memory as the type given in
    /// `dtype`, or null if not applicable.
    pub ptr: [AtomicPtr<c_void>; MAX_DEVICES],

    /// The size of this tensor in bytes.
    pub size_in_bytes: usize,

    /// The size of this tensor in the number of elements.
    pub size_in_elements: usize,

    /// The scale of this tensor,
    pub scale: f32
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            if !self.host.is_null() {
                cuda::cudaFreeHost(self.host);
            }

            for i in 0..MAX_DEVICES {
                let ptr = self.ptr[i].load(Ordering::Relaxed);

                if !ptr.is_null() {
                    cuda::cudaFree(ptr);
                }
            }
        }
    }
}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor {
            host: ptr::null_mut(),
            ptr: [
                AtomicPtr::new(ptr::null_mut()), AtomicPtr::new(ptr::null_mut()),
                AtomicPtr::new(ptr::null_mut()), AtomicPtr::new(ptr::null_mut()),
                AtomicPtr::new(ptr::null_mut()), AtomicPtr::new(ptr::null_mut()),
                AtomicPtr::new(ptr::null_mut()), AtomicPtr::new(ptr::null_mut()),
            ],
            size_in_bytes: 0,
            size_in_elements: 0,
            scale: 1.0
        }
    }
}

impl Tensor {
    pub fn get(&self, device_id: i32) -> *mut c_void {
        self.ptr[device_id as usize].load(Ordering::Relaxed)
    }

    pub fn set_host<T: Sized>(&mut self, data: Vec<T>) -> Result<(), Error> {
        unsafe {
            if !self.host.is_null() {
                check!(cuda::cudaFreeHost(self.host))?;
            }

            self.size_in_bytes = size_of::<T>() * data.len();
            self.size_in_elements = data.len();

            check!(cuda::cudaMallocHost(&mut self.host, self.size_in_bytes))?;

            ptr::copy_nonoverlapping(
                data.as_ptr() as *const c_void,
                self.host,
                self.size_in_bytes
            );

            Ok(())
        }
    }

    pub unsafe fn as_i32(&self) -> i32 {
        *(self.host as *const i32)
    }

    pub unsafe fn copy_to_device(&self, device_id: i32, stream: cuda::Stream) -> Result<bool, Error> {
        let device_id = device_id as usize;

        if self.ptr[device_id].load(Ordering::Relaxed).is_null() {
            let mut ptr = ptr::null_mut();
            let padded_size_in_bytes = if self.size_in_bytes % 32 == 0 {
                self.size_in_bytes
            } else {
                self.size_in_bytes + (32 - self.size_in_bytes % 32)
            };

            check!(cuda::cudaMalloc(&mut ptr, padded_size_in_bytes))?;
            check!(cuda::cudaMemcpyAsync(
                ptr,
                self.host,
                self.size_in_bytes,
                cuda::MemcpyKind::HostToDevice,
                stream
            ))?;

            if !self.ptr[device_id].compare_and_swap(ptr::null_mut(), ptr, Ordering::SeqCst).is_null() {
                check!(cuda::cudaStreamSynchronize(stream))?;  // wait for copy
                check!(cuda::cudaFree(ptr))?;

                Ok(false)
            } else {
                Ok(true)
            }
        } else {
            Ok(false)
        }
    }
}
