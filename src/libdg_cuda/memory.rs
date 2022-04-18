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

use crate::error::{Error, cudaError_t};
use crate::stream::{Stream, cudaStream_t};

use std::mem::size_of;
use std::ptr;
use libc::{c_void, size_t};

#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum cudaMemcpyKind_t {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3
}

#[link(name = "cudart")]
extern {
    fn cudaFree(dev_ptr: *const c_void) -> cudaError_t;
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: size_t) -> cudaError_t;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: size_t, kind: cudaMemcpyKind_t, stream: cudaStream_t) -> cudaError_t;
}

pub struct Ptr {
    dev_ptr: *mut c_void,
    size_in_bytes: usize
}

unsafe impl Send for Ptr {}

impl Default for Ptr {
    fn default() -> Self {
        Self { dev_ptr: ptr::null_mut(), size_in_bytes: 0 }
    }
}

impl Drop for Ptr {
    fn drop(&mut self) {
        unsafe { cudaFree(self.dev_ptr); }
    }
}

impl Ptr {
    pub fn new(size_in_bytes: usize) -> Result<Self, Error> {
        let mut dev_ptr = ptr::null_mut();
        let error = unsafe {
            cudaMalloc(&mut dev_ptr, size_in_bytes)
        };

        error.into_result(Self { dev_ptr, size_in_bytes })
    }

    pub fn from_slice<T: Sized>(data: &[T], stream: &Stream) -> Result<Self, Error> {
        let mut ptr = Self::new(size_of::<T>() * data.len())?;
        ptr.copy_from_slice(data, stream)?;

        Ok(ptr)
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.dev_ptr
    }

    pub fn is_null(&self) -> bool {
        self.dev_ptr.is_null()
    }

    pub fn resize(&mut self, size_in_bytes: usize) -> Result<(), Error> {
        *self = Self::new(size_in_bytes)?;
        Ok(())
    }

    pub fn copy_from_slice<T: Sized>(&mut self, data: &[T], stream: &Stream) -> Result<(), Error> {
        debug_assert!(size_of::<T>() * data.len() <= self.size_in_bytes);

        let size_in_bytes = size_of::<T>() * data.len();
        let error = unsafe {
            cudaMemcpyAsync(
                self.dev_ptr,
                data.as_ptr() as *const _,
                size_in_bytes,
                cudaMemcpyKind_t::HostToDevice,
                **stream
            )
        };

        error.into_result(())
    }

    pub fn copy_from_slice_offset<T: Sized>(&mut self, offset_in_bytes: usize, data: &[T], stream: &Stream) -> Result<(), Error> {
        debug_assert!(size_of::<T>() * data.len() + offset_in_bytes <= self.size_in_bytes);

        let size_in_bytes = size_of::<T>() * data.len();
        let error = unsafe {
            cudaMemcpyAsync(
                self.dev_ptr.offset(offset_in_bytes as isize),
                data.as_ptr() as *const _,
                size_in_bytes,
                cudaMemcpyKind_t::HostToDevice,
                **stream
            )
        };

        error.into_result(())
    }

    pub fn copy_from_ptr(&mut self, src: &Ptr, size_in_bytes: usize, stream: &Stream) -> Result<(), Error> {
        debug_assert!(size_in_bytes <= self.size_in_bytes);

        let error = unsafe {
            cudaMemcpyAsync(
                self.dev_ptr,
                src.as_ptr(),
                size_in_bytes,
                cudaMemcpyKind_t::DeviceToDevice,
                **stream
            )
        };

        error.into_result(())
    }

    pub fn size_in_bytes(&self) -> usize {
        self.size_in_bytes
    }

    pub fn try_clone(&self, stream: &Stream) -> Result<Ptr, Error> {
        let other = Self::new(self.size_in_bytes)?;
        let error = unsafe {
            cudaMemcpyAsync(
                other.dev_ptr,
                self.dev_ptr,
                self.size_in_bytes,
                cudaMemcpyKind_t::DeviceToDevice,
                **stream
            )
        };

        error.into_result(other)
    }

    pub fn to_vec<T: Sized + Clone + Default>(&self, stream: &Stream) -> Result<Vec<T>, Error> {
        let len = self.size_in_bytes / size_of::<T>();
        let mut out = vec! [T::default(); len];
        let error = unsafe {
            cudaMemcpyAsync(
                out.as_mut_ptr() as *mut _,
                self.dev_ptr as *const _,
                self.size_in_bytes,
                cudaMemcpyKind_t::DeviceToHost,
                **stream
            )
        };

        stream.synchronize()?;
        error.into_result(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_slice_to_vec() {
        let ptr = Ptr::from_slice(&[1, 2, 3, 4], &Stream::default()).unwrap();

        assert_eq!(ptr.to_vec::<i32>(&Stream::default()).unwrap(), vec! [1, 2, 3, 4]);
    }

    #[test]
    fn try_clone() {
        let ptr = Ptr::from_slice(&[1, 2, 3, 4], &Stream::default()).unwrap();
        let other = ptr.try_clone(&Stream::default()).unwrap();

        assert_eq!(
            ptr.to_vec::<i32>(&Stream::default()).unwrap(),
            other.to_vec::<i32>(&Stream::default()).unwrap()
        );
    }
}
