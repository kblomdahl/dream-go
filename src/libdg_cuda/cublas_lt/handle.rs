// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use crate::cublas_lt::{Status, cublasStatus_t};

use libc::c_void;
use std::ops::Deref;
use std::ptr;

#[allow(non_camel_case_types)]
pub type cublasLtHandle_t = *const c_void;

#[link(name = "cublasLt")]
extern {
    fn cublasLtCreate(light_handle: *mut cublasLtHandle_t) -> cublasStatus_t;
    fn cublasLtDestroy(light_handle: cublasLtHandle_t) -> cublasStatus_t;
}

pub struct Handle {
    handle: cublasLtHandle_t
}

unsafe impl Send for Handle {}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { cublasLtDestroy(self.handle) };
    }
}

impl Handle {
    pub fn new() -> Result<Handle, Status> {
        let mut out = Self { handle: ptr::null_mut() };
        let status = unsafe { cublasLtCreate(&mut out.handle) };

        status.into_result(out)
    }
}

impl Deref for Handle {
    type Target = cublasLtHandle_t;

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
