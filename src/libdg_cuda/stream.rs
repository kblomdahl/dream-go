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

use std::ptr;
use std::ops::Deref;
use libc::{c_void, c_uint};

#[allow(non_camel_case_types)]
pub type cudaEvent_t = *const c_void;

#[allow(non_camel_case_types)]
pub type cudaStream_t = *const c_void;

#[link(name = "cuda")]
#[link(name = "cudart")]
extern {
    pub fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> Error;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> Error;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> Error;

    pub fn cudaStreamCreateWithFlags(stream: *mut cudaStream_t, flags: c_uint) -> Error;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> Error;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> Error;
    pub fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) -> Error;
    pub fn cudaStreamQuery(stream: cudaStream_t) -> Error;
}

/// CUDA event types
pub struct Event {
    event: cudaEvent_t
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { cudaEventDestroy(self.event) };
    }
}

impl Event {
    pub fn new() -> Result<Self, Error> {
        let mut out = Self { event: ptr::null_mut() };
        let status = unsafe { cudaEventCreateWithFlags(&mut out.event, 2) };

        status.into_result(out)
    }

    pub fn record(&self, stream: &Stream) -> Result<(), Error> {
        unsafe { cudaEventRecord(self.event, stream.stream) }.into_result(())
    }
}

/// CUDA stream
pub struct Stream {
    stream: cudaStream_t
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { cudaStreamDestroy(self.stream) };
    }
}

impl Stream {
    pub fn new() -> Result<Self, Error> {
        let mut out = Self { stream: ptr::null_mut() };
        let status = unsafe { cudaStreamCreateWithFlags(&mut out.stream, 1) };

        status.into_result(out)
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        unsafe { cudaStreamSynchronize(self.stream) }.into_result(())
    }

    pub fn query(&self) -> Result<(), Error> {
        unsafe { cudaStreamQuery(self.stream) }.into_result(())
    }

    pub fn wait_event(&self, event: &Event) -> Result<(), Error> {
        unsafe { cudaStreamWaitEvent(self.stream, event.event, 0) }.into_result(())
    }
}

impl Deref for Stream {
    type Target = cudaStream_t;

    fn deref(&self) -> &Self::Target {
        &self.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_event() {
        assert!(Event::new().is_ok());
    }

    #[test]
    fn can_create_stream() {
        assert!(Stream::new().is_ok());
    }
}
