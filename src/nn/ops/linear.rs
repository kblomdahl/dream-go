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

use nn::ffi::cublas;
use nn::ffi::cuda;
use nn::ffi::cudnn;
use nn::ops::tensor::*;
use util::types::*;

/// Operator for performing a matrix multiplication using cuDNN.
pub struct Linear {
    pub input_nchw: cudnn::TensorDescriptor,
    pub workspace_size: usize,

    pub alpha: f32,
    pub beta: f32,

    // same as `alpha` and `beta` but in half precision.
    pub halpha: f16,
    pub hbeta: f16,

    /// The scale to use for the output tensor when adding bias
    pub delta_1: f32,

    /// The scale to use for the bias tensor when adding bias
    pub delta_2: f32
}

impl Drop for Linear {
    fn drop(&mut self) {
        unsafe {
            check!(cudnn::cudnnDestroyTensorDescriptor(self.input_nchw));
        }
    }
}

impl Linear {
    /// Returns a new output tensor that has the correct type, shape, and scale
    /// for the output of the operator `weights * input + offset`.
    /// 
    /// # Arguments
    /// 
    /// * `input` -
    /// * `k` -
    /// * `c` -
    /// * `weights` -
    /// * `offset` -
    /// 
    pub fn calibrate(
        input: &Tensor,
        k: i32,
        c: i32,
        weights: &Tensor,
        offset: &Tensor
    ) -> Tensor
    {
        let output = Tensor::default();

        weights.set_data_type_like(input);
        weights.set_shape(vec! [k, c]);

        offset.set_data_type_like(input);
        offset.set_scale(1.0);
        offset.set_shape(vec![1, k, 1, 1]);

        output.set_scale(1.0);
        output.set_data_type(input.get_data_type(), cudnn::TensorFormat::NCHW);
        output.set_shape(vec! [input.get_shape()[0], k, 1, 1]);
        output
    }

    /// Returns a linear operator that perform the following operation
    /// `output = weights * input + offset`, but does not blend the input with
    /// the output.
    /// 
    /// # Arguments
    /// 
    /// * `input` -
    /// * `weights` -
    /// * `offset` -
    /// * `output` -
    /// 
    pub fn new(
        input: &Tensor,
        weights: &Tensor,
        offset: &Tensor,
        output: &Tensor,
    ) -> Linear
    {
        let alpha = input.get_scale() * weights.get_scale() / output.get_scale();
        let (input_nchw, workspace_size) = if input.get_format() == cudnn::TensorFormat::NHWC {
            let input_nchw = Tensor::zeros_like(input);
            input_nchw.set_data_type(input.get_data_type(), cudnn::TensorFormat::NCHW);

            unsafe {
                (input_nchw.get_tensor_descriptor(), input_nchw.get_size_in_bytes())
            }
        } else {
            (ptr::null(), 0)
        };

        Linear {
            input_nchw: input_nchw,
            workspace_size: workspace_size,

            alpha: alpha,
            halpha: f16::from(alpha),

            beta: 0.0,
            hbeta: f16::from(0.0),

            delta_1: 1.0,
            delta_2: offset.get_scale() / output.get_scale()
        }
    }

    /// Performs the appropriate cuDNN calls to compute the matrix multiplication and
    /// stores the result in `output_data`. The workspace is used if we need to transform
    /// the tensor from `NHWC` to `NCHW`.
    /// 
    /// # Arguments
    /// 
    /// * `handle_blas` - 
    /// * `handle_dnn` - 
    /// * `input` - 
    /// * `input_data` - 
    /// * `batch_size` - 
    /// * `k` - 
    /// * `c` - 
    /// * `weights` - 
    /// * `offset` - 
    /// * `output` - 
    /// * `output_data` - 
    /// * `workspace_data` - 
    /// * `workspace_size` - 
    /// 
    pub fn forward(
        &self,
        handle_blas: cublas::Handle,
        handle_dnn: cudnn::Handle,
        input: &Tensor,
        input_data: *const c_void,
        batch_size: i32,
        k: i32,
        c: i32,
        weights: &Tensor,
        offset: &Tensor,
        output: &Tensor,
        output_data: *mut c_void,
        workspace_data: *mut c_void,
        workspace_size: usize
    )
    {
        unsafe {
            // if necessary transform the input tensor to NCHW so that we get
            // the correct dimensions
            let input_data = if input.get_format() == cudnn::TensorFormat::NHWC {
                const ONE: f32 = 1.0;
                const ZERO: f32 = 0.0;

                assert!(workspace_size >= input.get_size_in_bytes());
                check!(cudnn::cudnnTransformTensor(
                    handle_dnn,
                    &ONE, input.tensor_desc, input_data,
                    &ZERO, self.input_nchw, workspace_data
                ));

                workspace_data as *const c_void
            } else {
                input_data
            };

            // 
            let (algo, compute_type) = if cfg!(feature = "tensor-core") {
                (cublas::GemmAlgo::DfaltTensorOp, cuda::DataType::R32F)
            } else {
                (cublas::GemmAlgo::Dfalt, output.get_data_type().to_cuda())
            };
            let (alpha, beta) = match compute_type {
                cuda::DataType::R32F => (&self.alpha as *const f32 as *const c_void, &self.beta as *const f32 as *const c_void),
                cuda::DataType::R16F => (&self.halpha as *const f16 as *const c_void, &self.hbeta as *const f16 as *const c_void),
                _ => panic!()
            };

            check!(cublas::cublasGemmEx(
                handle_blas,
                cublas::Operation::N,
                cublas::Operation::N,
                k, batch_size, c,  // output_dims, batch_size, input_dims
                alpha,  // alpha
                weights.ptr, weights.get_data_type().to_cuda(), k,  // input_2
                input_data, input.get_data_type().to_cuda(), c,  // input_1
                beta,  // beta
                output_data, output.get_data_type().to_cuda(), k,  // output
                compute_type,
                algo
            ));

            check!(cudnn::cudnnAddTensor(
                handle_dnn,
                &self.delta_2,  // alpha
                offset.tensor_desc, offset.ptr,  // bias
                &self.delta_1,  // beta
                output.tensor_desc, output_data  // input and output
            ));
        }
    }
}
