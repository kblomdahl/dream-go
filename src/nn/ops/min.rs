// Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use libc::c_void;
use std::ptr;

use nn::ffi::{cuda, cudnn};
use nn::ops::tensor::*;
use util::types::f16;

/// Operator for computing `min(A, x)` where `x` is a constant.
pub struct Min {
    pub alpha1: f32,
    pub alpha2: f32,
    pub beta: f32,

    pub descr: cudnn::OpTensorDescriptor,
    pub constant: cudnn::TensorDescriptor,
    pub constant_data: *mut c_void
}

impl Drop for Min {
    fn drop(&mut self) {
        unsafe {
            cudnn::cudnnDestroyOpTensorDescriptor(self.descr);
            cudnn::cudnnDestroyTensorDescriptor(self.constant);
            cuda::cudaFree(self.constant_data);
        }
    }
}

impl Min {
    /// Returns a new minimum operator that mutate `input` in-place to not
    /// contain any value greater than `alpha`.
    /// 
    /// # Arguments
    /// 
    /// * `input` -
    /// * `alpha` -
    /// 
    pub fn new(input: &Tensor, alpha: f32) -> Min {
        let mut desc = ptr::null();
        let mut constant = ptr::null();
        let mut constant_data = ptr::null_mut();
        let data_type = if input.get_data_type().is_floating_point() {
            input.get_data_type()
        } else {
            cudnn::DataType::Float
        };

        unsafe {
            check!(cudnn::cudnnCreateOpTensorDescriptor(&mut desc));
            check!(cudnn::cudnnSetOpTensorDescriptor(
                desc,
                cudnn::OpTensorOp::Min,
                data_type,
                cudnn::NanPropagation::NotPropagateNan
            ));

            check!(cudnn::cudnnCreateTensorDescriptor(&mut constant));
            check!(cudnn::cudnnSetTensor4dDescriptor(
                constant,
                input.get_format(),
                data_type,
                1, 1, 1, 1
            ));

            check!(cuda::cudaMalloc(&mut constant_data, 4));

            if data_type == cudnn::DataType::Float {
                let one: f32 = 1.0;

                check!(cuda::cudaMemcpy(
                    constant_data,
                    &one as *const f32 as *const c_void,
                    4,
                    cuda::MemcpyKind::HostToDevice
                ));
            } else if data_type == cudnn::DataType::Half {
                let one: f16 = f16::from(1.0);

                check!(cuda::cudaMemcpy(
                    constant_data,
                    &one as *const _ as *const c_void,
                    2,
                    cuda::MemcpyKind::HostToDevice
                ));
            } else {
                unimplemented!();
            }
        }

        Min {
            alpha1: 1.0,
            alpha2: alpha / input.get_scale(),
            beta: 0.0,

            descr: desc,
            constant: constant,
            constant_data: constant_data
        }
    }

    /// Perform the appropriate cuDNN calls to perform the min operation on
    /// the `input_data`.
    /// 
    /// # Arguments
    /// 
    /// * `handle` -
    /// * `input` -
    /// * `input_data` -
    /// * `output` -
    /// * `output_data` -
    /// 
    pub fn forward(
        &self,
        handle: cudnn::Handle,
        input: &Tensor,
        input_data: *const c_void
    )
    {
        unsafe {
            check!(cudnn::cudnnOpTensor(
                handle,
                self.descr,
                &self.alpha1, input.tensor_desc, input_data,
                &self.alpha2, self.constant, self.constant_data,
                &self.beta, input.tensor_desc, input_data
            ));
        }
    }
}
