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
use nn::cuda::Stream;

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

    /// This function sets the cuBLAS library stream, which will be used to execute all
    /// subsequent calls to the cuBLAS library functions. If the cuBLAS library stream
    /// is not set, all kernels use the defaultNULL stream. In particular, this routine
    /// can be used to change the stream between kernel launches and then to reset the
    /// cuBLAS library stream back to NULL.
    /// 
    /// # Arguments
    /// 
    /// * `handle` - handle to the cuBLAS library context.
    /// * `stream` -
    /// 
    pub fn cublasSetStream_v2(handle: Handle, stream: Stream) -> Status;

    /// This function performs the matrix-matrix multiplication
    /// 
    /// ```C = α op(A) op(B) + β C```
    /// 
    /// where `α` and `β` are scalars, and `A`, `B` and `C` are matrices stored in
    /// column-major format with dimensions `op(A)` m × k , `op(B)` k × n
    /// and `C` m × n , respectively. Also, for matrix `A`
    /// 
    /// ```op(A) = A   if transa == CUBLAS_OP_N```
    /// 
    /// ```        A^T if transa == CUBLAS_OP_T```
    /// 
    /// ```        A^H if transa == CUBLAS_OP_C.```
    /// 
    /// and `op(B)` is defined similarly for matrix B .
    /// 
    /// # Arguments
    /// 
    /// * `handle` - handle to the cuBLAS library context.
    /// * `transA` - operation `op(A)` that is non- or (conj.) transpose.
    /// * `transB` - operation `op(B)` that is non- or (conj.) transpose.
    /// * `m` - number of rows of matrix `op(A)` and `C`.
    /// * `n` - number of columns of matrix `op(B)` and `C`.
    /// * `k` - number of columns of `op(A)` and rows of `op(B)`.
    /// * `alpha` - scalar used for multiplication.
    /// * `A` - array of dimensions lda × k with `lda>=max(1,m)` if `transa == CUBLAS_OP_N` and lda × m with `lda>=max(1,k)` otherwise.
    /// * `lda` - leading dimension of two-dimensional array used to store the matrix `A`.
    /// * `B` - array of dimension ldb × n with `ldb>=max(1,k)` if `transa == CUBLAS_OP_N` and ldb × k with `ldb>=max(1,n)` otherwise.
    /// * `ldb` - leading dimension of two-dimensional array used to store matrix `B`.
    /// * `beta` - scalar used for multiplication.
    /// * `C` - array of dimensions ldc × n with `ldc>=max(1,m)`.
    /// * `ldc` - leading dimension of a two-dimensional array used to store the matrix `C`.
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
