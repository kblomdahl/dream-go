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

use nn::ffi::cudnn;
use nn::ops::tensor::*;
use nn::ops::min::*;

/// Operator for performing a 2D convolution using cuDNN.
pub struct Convolution {
    pub alpha: f32,
    pub beta: f32,

    pub descr: cudnn::ConvolutionDescriptor,
    pub fwd_algo: cudnn::ConvolutionFwdAlgo,
    pub workspace_size: usize,

    pub min_six: Min
}

impl Convolution {
    /// Set the correct type, shape, and scale for the operands of the
    /// operator `output = weights * input + offset`.
    ///
    /// # Arguments
    ///
    /// * `input` -
    /// * `k` -
    /// * `c` -
    /// * `h` -
    /// * `w` -
    /// * `weights` -
    /// * `offset` -
    /// * `output` -
    ///
    pub fn calibrate(
        input: &Tensor,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
        weights: &Tensor,
        offset: &Tensor,
        output: &Tensor
    )
    {
        // copy the shape from the input tensor to the output
        let input_shape = input.get_shape();

        output.set_shape(vec! [input_shape[0], k, input_shape[2], input_shape[3]]);

        // set the data type of the weights and offset to match the input tensor
        // in regard to both type and shape
        weights.set_data_type_like(input);
        weights.set_shape(vec! [k, c, h, w]);

        offset.set_data_type_like(input);
        offset.set_shape(vec! [1, k, 1, 1]);
        offset.set_scale(output.get_scale());

        // the `i8` data type requires the number of input and output channels
        // to be a multiple of four. If this is not the case then fallback to
        // using floating point for the output
        output.set_data_type({
            if input.get_data_type() == cudnn::DataType::Int8 && k % 4 != 0 {
                cudnn::DataType::Float
            } else {
                input.get_data_type()
            }
        }, input.get_format());

        // the `i8` data type requires the offset to be an `f32`, but scaled
        // to the range [-128,+127] as if it was an `i8`.
        if input.get_data_type() == cudnn::DataType::Int8 {
            offset.set_data_type(cudnn::DataType::Float, offset.get_format());
            offset.set_scale(offset.get_scale() / 127.0);
        }
    }

    /// Returns a convolutional operator that is optimized for the given
    /// arguments.
    ///
    /// # Arguments
    ///
    /// * `handle` -
    /// * `alpha` -
    /// * `input` -
    /// * `weights` -
    /// * `beta` -
    /// * `blend` -
    /// * `bias` -
    /// * `output` -
    ///
    pub fn new(
        handle: cudnn::Handle,
        mut alpha: f32,
        input: &Tensor,
        weights: &Tensor,
        beta: f32,
        blend: &Tensor,
        _bias: &Tensor,
        output: &Tensor,
    ) -> Convolution
    {
        let mut fwd_algo: cudnn::ConvolutionFwdAlgo = cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm;
        let mut workspace_size: usize = 0;
        let mut conv_desc = ptr::null();

        unsafe {
            let x_desc = input.get_tensor_descriptor();
            let w_desc = weights.filter_desc;
            let w_shape = weights.get_shape();
            let y_desc = output.get_tensor_descriptor();
            let compute_type = if input.get_data_type() == cudnn::DataType::Int8 {
                cudnn::DataType::Int32
            } else if cfg!(feature = "tensor-core") {
                cudnn::DataType::Float
            } else {
                input.get_data_type()
            };

            check!(cudnn::cudnnCreateConvolutionDescriptor(&mut conv_desc));
            check!(cudnn::cudnnSetConvolution2dDescriptor(
                conv_desc,
                w_shape[2] / 2, w_shape[3] / 2,
                1, 1, 1, 1,
                cudnn::ConvolutionMode::CrossCorrelation,
                compute_type
            ));

            #[cfg(feature = "tensor-core")]
            {
                check!(cudnn::cudnnSetConvolutionMathType(
                    conv_desc,
                    cudnn::MathType::TensorOpMath
                ));
            }

            if input.get_data_type() != cudnn::DataType::Int8 {
                // this fails with `NotSupported` for `q8`, but the default
                // algorithm works fine.
                check!(cudnn::cudnnGetConvolutionForwardAlgorithm(
                    handle,
                    x_desc,
                    w_desc,
                    conv_desc,
                    y_desc,
                    cudnn::ConvolutionFwdPreference::PreferFastest,
                    0,
                    &mut fwd_algo
                ));
            }

            check!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
                handle,
                x_desc,
                w_desc,
                conv_desc,
                y_desc,
                fwd_algo,
                &mut workspace_size
            ));

            check!(cudnn::cudnnDestroyTensorDescriptor(y_desc));
            check!(cudnn::cudnnDestroyTensorDescriptor(x_desc));
        }

        // both the weights and the input are scaled by `127` we we need
        // to cancel one of them out
        if input.get_data_type() == cudnn::DataType::Int8 {
            alpha /= 127.0;
        }

        Convolution {
            alpha: alpha * (input.get_scale() * weights.get_scale()) / output.get_scale(),
            beta: beta * (blend.get_scale() / output.get_scale()),

            descr: conv_desc,
            fwd_algo: fwd_algo,
            workspace_size: workspace_size,

            min_six: Min::new(output, 6.0)
        }
    }

    /// Performs the appropriate cuDNN calls to compute the convolution and
    /// stores the result in `output_data`.
    ///
    /// # Arguments
    ///
    /// * `handle` -
    /// * `input` -
    /// * `input_data` -
    /// * `weights` -
    /// * `blend` -
    /// * `blend_data` -
    /// * `weights` -
    /// * `offset` -
    /// * `output` -
    /// * `output_data` -
    /// * `workspace_data` -
    /// * `workspace_size` -
    /// * `relu` -
    ///
    pub fn forward(
        &self,
        handle: cudnn::Handle,
        input: &Tensor,
        input_data: *const c_void,
        weights: &Tensor,
        blend: &Tensor,
        blend_data: *const c_void,
        offset: &Tensor,
        output: &Tensor,
        output_data: *mut c_void,
        workspace_data: *mut c_void,
        workspace_size: usize,
        relu: cudnn::ActivationDescriptor,
    )
    {
        unsafe {
            check!(cudnn::cudnnConvolutionBiasActivationForward(
                handle,
                &self.alpha,
                input.tensor_desc, input_data,  // input
                weights.filter_desc, weights.ptr,  // weights
                self.descr,  // convolution
                self.fwd_algo,  // algo
                workspace_data, workspace_size,  // workspace
                &self.beta,  // beta
                blend.tensor_desc, blend_data,  // blend
                offset.tensor_desc, offset.ptr,  // bias
                relu,
                output.tensor_desc, output_data  // output
            ));

            if input.get_data_type() == cudnn::DataType::Int8 && output.get_data_type() == cudnn::DataType::Float
            {
                // we have to re-scale the output tensor from [-128,+127] to [0,1] after
                // the convolution because it uses an `i32` accumulator which cannot store
                // fractions.
                const C_127: f32 = 0.00787401575;

                check!(cudnn::cudnnScaleTensor(
                    handle,
                    output.tensor_desc, output_data,
                    &C_127
                ));
            }

            if output.get_data_type() == cudnn::DataType::Float {
                self.min_six.forward(handle, output, output_data);
            } else {
                debug_assert!(output.get_scale() <= 6.0);
            }
        }
    }
}

impl Drop for Convolution {
    fn drop(&mut self) {
        unsafe {
            cudnn::cudnnDestroyConvolutionDescriptor(self.descr);
        }
    }
}
