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
use std::ptr::{null_mut, Unique};

use libc::{c_int, c_void, size_t};

use ::Error;
use ::stream::{CudaStream, Stream};

#[link(name = "cuda")]
#[link(name = "cudart")]
extern {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: size_t) -> c_int;
    fn cudaFree(dev_ptr: *mut c_void) -> c_int;

    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: size_t,
        kind: cudaMemcpyKind_t,
        stream: CudaStream
    ) -> c_int;
}

#[repr(i32)]
#[allow(non_camel_case_types)] pub enum cudaMemcpyKind_t {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

#[derive(Debug)]
pub struct Ptr(Option<Unique<c_void>>);

impl Drop for Ptr {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref x) = self.0 {
                cudaFree(x.as_ptr());
            }
        }
    }
}

impl Ptr {
    pub fn new(size_in_bytes: usize) -> Result<Ptr, Error> {
        assert!(size_in_bytes > 0);

        let mut dev_ptr = null_mut();
        let success = unsafe {
            cudaMalloc(&mut dev_ptr, size_in_bytes)
        };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(Ptr(Some(Unique::new(dev_ptr).unwrap())))
        }
    }

    pub fn null() -> Ptr {
        Ptr(None)
    }

    pub fn from_vec<T: Sized>(data: &Vec<T>, stream: &Stream) -> Result<Ptr, Error> {
        let size_in_bytes = size_of::<T>() * data.len();
        let ptr = Self::new(size_in_bytes)?;

        copy_nonoverlapping(
            data.as_ptr() as *const c_void,
            ptr.as_ptr(),
            size_in_bytes,
            cudaMemcpyKind_t::HostToDevice,
            stream
        )?;

        Ok(ptr)
    }

    pub fn as_ptr(&self) -> *mut c_void {
        match self.0 {
            None => null_mut(),
            Some(ref x) => x.as_ptr()
        }
    }
}

pub fn copy_nonoverlapping<T: Sized>(
    src: *const T,
    dst: *mut T,
    count: usize,
    kind: cudaMemcpyKind_t,
    stream: &Stream
) -> Result<(), Error>
{
    let success = unsafe {
        cudaMemcpyAsync(
            dst as *mut c_void,
            src as *const c_void,
            count * size_of::<T>(),
            kind,
            stream.as_ptr()
        )
    };

    if success != 0 {
        Err(Error::CudaError(success))
    } else {
        Ok(())
    }
}