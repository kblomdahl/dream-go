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
use crate as cuda;

use libc::{c_void, size_t};
use std::ops::Deref;
use std::ptr;

#[allow(non_camel_case_types)]
pub type cublasLtMatmulDesc_t = *const c_void;

#[allow(non_camel_case_types)]
#[repr(i32)]
enum cublasLtMatmulDescAttributes_t {
    Epilogue = 7,
    BiasPointer = 8,
    TransA = 3,
    TransB = 4
}

#[link(name = "cublasLt")]
extern {
    fn cublasLtMatmulDescCreate(
        matmulDesc: *mut cublasLtMatmulDesc_t,
        compute_type: cublasComputeType_t,
        scale_type: cudaDataType
    ) -> cublasStatus_t;
    fn cublasLtMatmulDescDestroy(matmulDesc: cublasLtMatmulDesc_t) -> cublasStatus_t;
    fn cublasLtMatmulDescSetAttribute(
        matmul_desc: cublasLtMatmulDesc_t,
        attr: cublasLtMatmulDescAttributes_t,
        buf: *const c_void,
        size_in_bytes: size_t
    ) -> cublasStatus_t;
}

pub struct MatmulDesc {
    desc: cublasLtMatmulDesc_t
}

unsafe impl Send for MatmulDesc {}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        unsafe { cublasLtMatmulDescDestroy(self.desc) };
    }
}

impl MatmulDesc {
    pub fn new(
        compute_type: ComputeType,
        scale_type: DataType
    ) -> Result<Self, Status>
    {
        let mut out = Self { desc: ptr::null_mut() };
        let status = unsafe {
            cublasLtMatmulDescCreate(
                &mut out.desc,
                compute_type,
                scale_type
            )
        };

        status.into_result(out)
    }

    pub fn with_epilogue(self, epilogue: Epilogue) -> Result<Self, Status> {
        let status = unsafe {
            cublasLtMatmulDescSetAttribute(
                self.desc,
                cublasLtMatmulDescAttributes_t::Epilogue,
                &epilogue as *const Epilogue as *const _,
                ::std::mem::size_of_val(&epilogue)
            )
        };

        status.into_result(self)
    }

    pub fn set_bias(&self, bias: &cuda::Ptr) -> Result<(), Status> {
        let bias_ptr = bias.as_ptr();
        let status = unsafe {
            cublasLtMatmulDescSetAttribute(
                self.desc,
                cublasLtMatmulDescAttributes_t::BiasPointer,
                &bias_ptr as *const _ as *const _,
                ::std::mem::size_of::<*const c_void>()
            )
        };

        status.into_result(())
    }

    pub fn with_transpose_a(self, op: Operation) -> Result<Self, Status> {
        let status = unsafe {
            cublasLtMatmulDescSetAttribute(
                self.desc,
                cublasLtMatmulDescAttributes_t::TransA,
                &op as *const _ as *const _,
                ::std::mem::size_of_val(&op)
            )
        };

        status.into_result(self)
    }

    pub fn with_transpose_b(self, op: Operation) -> Result<Self, Status> {
        let status = unsafe {
            cublasLtMatmulDescSetAttribute(
                self.desc,
                cublasLtMatmulDescAttributes_t::TransB,
                &op as *const _ as *const _,
                ::std::mem::size_of_val(&op)
            )
        };

        status.into_result(self)
    }
}

impl Deref for MatmulDesc {
    type Target = cublasLtMatmulDesc_t;

    fn deref(&self) -> &Self::Target {
        &self.desc
    }
}
