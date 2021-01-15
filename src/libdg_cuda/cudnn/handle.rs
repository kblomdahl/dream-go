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

use crate::cudnn::{Status, cudnnStatus_t};
use crate::Stream;

use libc::c_void;
use std::ops::Deref;
use std::ptr;

#[allow(non_camel_case_types)]
pub type cudnnHandle_t = *const c_void;

#[link(name = "cudnn_ops_infer")]
extern {
    fn cudnnCreate(handle: *const cudnnHandle_t) -> cudnnStatus_t;
    fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;
    fn cudnnSetStream(handle: cudnnHandle_t, stream: *const c_void) -> cudnnStatus_t;
}

/// `cudnnHandle_t` is a pointer to an opaque structure holding the cuDNN
/// library context. The cuDNN library context must be created using
/// `cudnnCreate()` and the returned handle must be passed to all subsequent
/// library function calls. The context should be destroyed at the end using
/// `cudnnDestroy()`. The context is associated with only one GPU device, the
/// current device at the time of the call to cudnnCreate(). However, multiple
/// contexts can be created on the same GPU device.
pub struct Handle {
    handle: cudnnHandle_t
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { cudnnDestroy(self.handle) };
    }
}

impl Handle {
    pub fn new() -> Result<Handle, Status> {
        let mut out = Self { handle: ptr::null_mut() };
        let status = unsafe { cudnnCreate(&mut out.handle) };

        status.into_result(out)
    }

    pub fn set_stream(&self, stream: &Stream) -> Result<(), Status> {
        unsafe { cudnnSetStream(self.handle, **stream) }.into_result(())
    }
}

impl Deref for Handle {
    type Target = cudnnHandle_t;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_handle() {
        assert!(Handle::new().is_ok());
    }
}
