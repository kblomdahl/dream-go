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

use crate::cublas_lt::*;

use libc::{c_void, size_t};
use std::ops::Deref;
use std::ptr;

#[allow(non_camel_case_types)]
pub type cublasLtMatrixLayout_t = *const c_void;

#[allow(non_camel_case_types)]
#[repr(i32)]
enum cublasLtMatrixLayoutAttribute_t {
    LayoutOrder = 1
}

#[link(name = "cublasLt")]
extern {
    fn cublasLtMatrixLayoutCreate(
        matrix_layout: *mut cublasLtMatrixLayout_t,
        data_type: cudaDataType,
        rows: u64,
        cols: u64,
        ld: i64
    ) -> cublasStatus_t;
    fn cublasLtMatrixLayoutDestroy(matrix_layout: cublasLtMatrixLayout_t) -> cublasStatus_t;
    fn cublasLtMatrixLayoutSetAttribute(
        matrix_layout: cublasLtMatrixLayout_t,
        attr: cublasLtMatrixLayoutAttribute_t,
        buf: *const c_void,
        size_in_bytes: size_t
    ) -> cublasStatus_t;
}

pub struct MatrixLayout {
    matrix_layout: cublasLtMatrixLayout_t
}

unsafe impl Send for MatrixLayout {}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        unsafe { cublasLtMatrixLayoutDestroy(self.matrix_layout) };
    }
}

impl MatrixLayout {
    pub fn new(
        data_type: DataType,
        rows: u64,
        cols: u64,
        ld: i64
    ) -> Result<Self, Status>
    {
        let mut out = Self { matrix_layout: ptr::null_mut() };
        let status = unsafe {
            cublasLtMatrixLayoutCreate(
                &mut out.matrix_layout,
                data_type,
                rows,
                cols,
                ld
            )
        };

        status.into_result(out)
    }

    pub fn with_order(self, order: Order) -> Result<Self, Status> {
        let status = unsafe {
            cublasLtMatrixLayoutSetAttribute(
                self.matrix_layout,
                cublasLtMatrixLayoutAttribute_t::LayoutOrder,
                &order as *const Order as *const _,
                4
            )
        };

        status.into_result(self)
    }
}

impl Deref for MatrixLayout {
    type Target = cublasLtMatrixLayout_t;

    fn deref(&self) -> &Self::Target {
        &self.matrix_layout
    }
}
