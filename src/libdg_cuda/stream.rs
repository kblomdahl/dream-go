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

use super::error::Error;

use std::ptr::{null_mut, Unique};
use libc::{c_void, c_int, c_uint};


pub type CudaEvent = *mut c_void;
pub type CudaStream = *mut c_void;

#[link(name = "cuda")]
#[link(name = "cudart")]
extern {
    fn cudaEventCreateWithFlags(event: *mut CudaEvent, flags: c_uint) -> c_int;
    fn cudaEventDestroy(event: CudaEvent) -> c_int;
    fn cudaEventRecord(event: CudaEvent, stream: CudaStream) -> c_int;

    fn cudaStreamCreateWithFlags(stream: *mut CudaStream, flags: c_uint) -> c_int;
    fn cudaStreamDestroy(stream: CudaStream) -> c_int;
    fn cudaStreamSynchronize(stream: CudaStream) -> c_int;
    fn cudaStreamWaitEvent(stream: CudaStream, event: CudaEvent, flags: c_uint) -> c_int;
}

pub struct Event(Unique<c_void>);

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { cudaEventDestroy(self.as_ptr()); }
    }
}

pub struct Stream {
    stream: Option<Unique<c_void>>,
    owned: bool
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.owned && self.stream.is_some() {
            unsafe { cudaStreamDestroy(self.as_ptr()); }
        }
    }
}

impl Default for Stream {
    fn default() -> Stream {
        Stream {
            stream: None,
            owned: false,
        }
    }
}

impl Stream {
    pub fn new() -> Result<Stream, Error> {
        let mut stream = null_mut();
        let success = unsafe { cudaStreamCreateWithFlags(&mut stream, 0x1) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(Stream {
                stream: Some(Unique::new(stream).unwrap()),
                owned: true,
            })
        }
    }

    pub fn from(stream: CudaStream) -> Stream {
        if stream.is_null() {
            Self::default()
        } else {
            Stream {
                stream: Some(Unique::new(stream).unwrap()),
                owned: false,
            }
        }
    }

    pub fn wait_for(&self, event: Event) -> Result<(), Error> {
        let success = unsafe { cudaStreamWaitEvent(self.as_ptr(), event.as_ptr(), 0) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(())
        }
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        let success = unsafe { cudaStreamSynchronize(self.as_ptr()) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(())
        }
    }

    pub(super) fn as_ptr(&self) -> CudaStream {
        match self.stream {
            None => null_mut(),
            Some(ref x) => x.as_ptr()
        }
    }
}

impl Event {
    pub fn new() -> Result<Event, Error> {
        let mut event = null_mut();
        let success = unsafe { cudaEventCreateWithFlags(&mut event, 0x2) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(Event(Unique::new(event).unwrap()))
        }
    }

    pub fn record(&self, stream: Stream) -> Result<(), Error> {
        let success = unsafe { cudaEventRecord(self.as_ptr(), stream.as_ptr()) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(())
        }
    }

    pub(super) fn as_ptr(&self) -> CudaEvent {
        self.0.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    // pass
}
