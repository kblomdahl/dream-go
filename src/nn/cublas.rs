// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use libc::{c_float, c_int, c_void};

#[repr(i32)]
#[allow(dead_code)]
#[derive(PartialEq, Eq, Debug)]
pub enum Status {
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 3,
    InvalidValue = 7,
    ArchMismatch = 8,
    MappingError = 11,
    ExecutionFailed = 13,
    InternalError = 14,
    NotSupported = 15,
    LicenseError = 16
}

#[repr(i32)]
#[allow(dead_code)]
pub enum Operation {
    N = 0,
    T = 1,
    C = 2
}

pub type Handle = *const c_void;

#[link(name = "cublas")]
extern {
    pub fn cublasCreate_v2(handle: *mut Handle) -> Status;
    pub fn cublasDestroy_v2(handle: Handle) -> Status;
    pub fn cublasSgemm_v2(
        handle: Handle,
        transA: Operation,
        transB: Operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_float,
        A: *const c_void,
        lda: c_int,
        B: *const c_void,
        ldb: c_int,
        beta: *const c_float,
        C: *mut c_void,
        ldc: c_int
    ) -> Status;
}
