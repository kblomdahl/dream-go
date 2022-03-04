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
pub type cublasLtMatmulPreference_t = *const c_void;

#[link(name = "cublasLt")]
extern {
    fn cublasLtMatmulPreferenceCreate(pref: *mut cublasLtMatmulPreference_t) -> cublasStatus_t;
    fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> cublasStatus_t;
}

pub struct MatmulPreference {
    pref: cublasLtMatmulPreference_t
}

unsafe impl Send for MatmulPreference {}

impl Drop for MatmulPreference {
    fn drop(&mut self) {
        unsafe { cublasLtMatmulPreferenceDestroy(self.pref) };
    }
}

impl MatmulPreference {
    pub fn new() -> Result<Self, Status> {
        let mut out = Self { pref: ptr::null_mut() };
        let status = unsafe { cublasLtMatmulPreferenceCreate(&mut out.pref) };

        status.into_result(out)
    }
}

impl Deref for MatmulPreference {
    type Target = cublasLtMatmulPreference_t;

    fn deref(&self) -> &Self::Target {
        &self.pref
    }
}
