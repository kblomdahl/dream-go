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
    DataType = 0,
    LayoutOrder = 1,
    Rows = 2,
    Cols = 3,
    Ld = 4
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
    fn cublasLtMatrixLayoutGetAttribute(
        matrix_layout: cublasLtMatrixLayout_t,
        attr: cublasLtMatrixLayoutAttribute_t,
        buf: *mut c_void,
        size_in_bytes: size_t,
        size_written: *mut size_t
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

    pub fn data_type(&self) -> Result<DataType, Status> {
        let mut data_type: DataType = DataType::Real32F;
        let mut size_written = 0;
        let status = unsafe {
            cublasLtMatrixLayoutGetAttribute(
                self.matrix_layout,
                cublasLtMatrixLayoutAttribute_t::DataType,
                &mut data_type as *mut _ as *mut _,
                ::std::mem::size_of_val(&data_type),
                &mut size_written as *mut _ as *mut _
            )
        };

        status.into_result(data_type)
    }

    pub fn rows(&self) -> Result<usize, Status> {
        let mut rows: u64 = 0;
        let mut size_written = 0;
        let status = unsafe {
            cublasLtMatrixLayoutGetAttribute(
                self.matrix_layout,
                cublasLtMatrixLayoutAttribute_t::Rows,
                &mut rows as *mut _ as *mut _,
                ::std::mem::size_of_val(&rows),
                &mut size_written as *mut _ as *mut _
            )
        };
        assert_eq!(size_written, ::std::mem::size_of_val(&rows));

        status.into_result(rows as usize)
    }

    pub fn cols(&self) -> Result<usize, Status> {
        let mut cols: u64 = 0;
        let mut size_written = 0;
        let status = unsafe {
            cublasLtMatrixLayoutGetAttribute(
                self.matrix_layout,
                cublasLtMatrixLayoutAttribute_t::Cols,
                &mut cols as *mut _ as *mut _,
                ::std::mem::size_of_val(&cols),
                &mut size_written as *mut _ as *mut _
            )
        };
        assert_eq!(size_written, ::std::mem::size_of_val(&cols));

        status.into_result(cols as usize)
    }

    pub fn ld(&self) -> Result<usize, Status> {
        let mut ld: i64 = 0;
        let mut size_written = 0;
        let status = unsafe {
            cublasLtMatrixLayoutGetAttribute(
                self.matrix_layout,
                cublasLtMatrixLayoutAttribute_t::Ld,
                &mut ld as *mut _ as *mut _,
                ::std::mem::size_of_val(&ld),
                &mut size_written as *mut _ as *mut _
            )
        };
        assert_eq!(size_written, ::std::mem::size_of_val(&ld));

        status.into_result(ld as usize)
    }

    pub fn size_in_bytes(&self) -> Result<usize, Status> {
        Ok(self.cols()? * self.ld()? * self.data_type()?.size_in_bytes())
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
