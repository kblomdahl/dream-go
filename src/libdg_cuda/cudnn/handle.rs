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

use super::*;
use Error;
use Stream;

use libc::{c_void, c_int};
use std::ptr::{null_mut, Unique};

#[link(name = "cudnn")]
extern {
    fn cudnnCreate(handle: *mut cudnnHandle_t) -> c_int;
    fn cudnnDestroy(handle: cudnnHandle_t) -> c_int;
    fn cudnnSetStream(handle: cudnnHandle_t, stream: *mut c_void) -> c_int;
    fn cudnnGetStream(handle: cudnnHandle_t, stream: *mut *mut c_void) -> c_int;
}

#[derive(Debug)]
pub struct Handle(Unique<c_void>);

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { cudnnDestroy(self.as_ptr()) };
    }
}

impl Handle {
    pub fn new() -> Result<Handle, Error> {
        let mut handle = null_mut();
        let success = unsafe { cudnnCreate(&mut handle) };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        Ok(Handle(Unique::new(handle).unwrap()))
    }

    pub fn set_stream(&mut self, stream: &Stream) -> Result<(), Error> {
        let success = unsafe { cudnnSetStream(self.as_ptr(), stream.as_ptr()) };

        if success != 0 {
            Err(Error::CudnnError(success))
        } else {
            Ok(())
        }
    }

    pub fn get_stream(&self) -> Result<Stream, Error> {
        let mut stream = null_mut();
        let success = unsafe {
            cudnnGetStream(self.as_ptr(), &mut stream as *mut _)
        };

        if success != 0 {
            Err(Error::CudnnError(success))
        } else {
            Ok(Stream::from(stream))
        }
    }

    pub(super) fn as_ptr(&self) -> cudnnHandle_t {
        self.0.as_ptr()
    }
}

