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

use super::*;
use Error;

use std::ptr::{null_mut, Unique};
use libc::c_void;

const ONE: f32 = 1.0;
const ZERO: f32 = 0.0;

#[link(name = "cudnn")]
extern {
    fn cudnnCreateOpTensorDescriptor(op_tensor_desc: *mut cudnnOpTensorDescriptor_t) -> c_int;
    fn cudnnDestroyOpTensorDescriptor(op_tensor_desc: cudnnOpTensorDescriptor_t) -> c_int;

    fn cudnnSetOpTensorDescriptor(
        op_tensor_desc: cudnnOpTensorDescriptor_t,
        op_tensor_op: cudnnOpTensorOp_t,
        op_tensor_comp_type: cudnnDataType_t,
        op_tensor_nan_opt: cudnnNanPropagation_t
    ) -> c_int;

    fn cudnnOpTensor(
        handle: cudnnHandle_t,
        op_tensor_desc: cudnnOpTensorDescriptor_t,
        alpha_1: *const c_void,
        a_desc: cudnnTensorDescriptor_t,
        a: *const c_void,
        alpha_2: *const c_void,
        b_desc: cudnnTensorDescriptor_t,
        b: *const c_void,
        beta: *const c_void,
        c_desc: cudnnTensorDescriptor_t,
        c: *const c_void,
    ) -> c_int;
}

#[derive(Debug)]
pub struct OpTensor(Unique<c_void>);

impl Drop for OpTensor {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyOpTensorDescriptor(self.as_ptr());
        }
    }
}

impl OpTensor {
    pub fn new(
        op_tensor_op: cudnnOpTensorOp_t,
        op_tensor_comp_type: cudnnDataType_t,
        op_tensor_nan_opt: cudnnNanPropagation_t
    ) -> Result<OpTensor, Error>
    {
        let mut or_tensor_desc = null_mut();
        let success = unsafe { cudnnCreateOpTensorDescriptor(&mut or_tensor_desc) };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        let success = unsafe {
            cudnnSetOpTensorDescriptor(
                or_tensor_desc,
                op_tensor_op,
                op_tensor_comp_type,
                op_tensor_nan_opt
            )
        };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        Ok(OpTensor(Unique::new(or_tensor_desc).unwrap()))
    }

    pub fn forward(
        &self,
        handle: &Handle,
        a: &Tensor,
        a_data: *const c_void,
        b: &Tensor,
        b_data: *const c_void,
        c: &Tensor,
        c_data: *mut c_void
    ) -> Result<(), Error>
    {
        let success = unsafe {
            cudnnOpTensor(
                handle.as_ptr(),
                self.as_ptr(),
                &ONE as *const _ as *const c_void,
                a.as_ptr(), a_data,
                &ONE as *const _ as *const c_void,
                b.as_ptr(), b_data,
                &ZERO as *const _ as *const c_void,
                c.as_ptr(), c_data,
            )
        };

        if success != 0 {
            Err(Error::CudnnError(success))
        } else {
            Ok(())
        }
    }

    pub(super) fn as_ptr(&self) -> cudnnOpTensorDescriptor_t {
        self.0.as_ptr()
    }
}