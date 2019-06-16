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

use libc::{c_void, c_int, size_t};
use std::ptr::{null_mut, Unique};

const ONE: f32 = 1.0;
const ZERO: f32 = 0.0;

#[link(name = "cudnn")]
extern {
    fn cudnnCreateConvolutionDescriptor(conv_desc: *mut cudnnConvolutionDescriptor_t) -> c_int;
    fn cudnnDestroyConvolutionDescriptor(conv_desc: cudnnConvolutionDescriptor_t) -> c_int;

    fn cudnnGetConvolutionForwardAlgorithm_v7(
        handle: cudnnHandle_t,
        x_desc: cudnnTensorDescriptor_t,
        w_desc: cudnnFilterDescriptor_t,
        conv_desc: cudnnConvolutionDescriptor_t,
        y_desc: cudnnTensorDescriptor_t,
        requested_algo_count: c_int,
        returned_algo_count: *mut c_int,
        perf_results: *mut cudnnConvolutionFwdAlgoPerf_t
    ) -> c_int;

    fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: cudnnHandle_t,
        x_desc: cudnnTensorDescriptor_t,
        w_desc: cudnnFilterDescriptor_t,
        conv_desc: cudnnConvolutionDescriptor_t,
        y_desc: cudnnTensorDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        size_in_bytes: *mut size_t
    ) -> c_int;

    fn cudnnSetConvolution2dDescriptor(
        conv_desc: cudnnConvolutionDescriptor_t,
        pad_h: c_int,
        pad_w: c_int,
        u: c_int,
        v: c_int,
        dilation_h: c_int,
        dilation_w: c_int,
        mode: cudnnConvolutionMode_t,
        compute_type: cudnnDataType_t
    ) -> c_int;

    fn cudnnSetConvolutionGroupCount(
        conv_desc: cudnnConvolutionDescriptor_t,
        group_count: c_int
    ) -> c_int;

    fn cudnnSetConvolutionMathType(
        conv_desc: cudnnConvolutionDescriptor_t,
        math_type: cudnnMathType_t
    ) -> c_int;

    fn cudnnConvolutionBiasActivationForward(
        handle: cudnnHandle_t,
        alpha_1: *const c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const c_void,
        w_desc: cudnnFilterDescriptor_t,
        w: *const c_void,
        conv_desc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        workspace: *mut c_void,
        workspace_size_in_bytes: size_t,
        alpha_2: *const c_void,
        z_desc: cudnnTensorDescriptor_t,
        z: *const c_void,
        bias_desc: cudnnTensorDescriptor_t,
        bias: *const c_void,
        activation_desc: cudnnActivationDescriptor_t,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut c_void
    ) -> c_int;
}

#[derive(Debug)]
pub struct Convolution(Unique<c_void>);

impl Drop for Convolution {
    fn drop(&mut self) {
        unsafe { cudnnDestroyConvolutionDescriptor(self.as_ptr()) };
    }
}

impl Convolution {
    pub fn new(filter: &Filter, group_count: usize) -> Result<Convolution, Error> {
        let mut conv_desc = null_mut();
        let success = unsafe { cudnnCreateConvolutionDescriptor(&mut conv_desc) };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        let ((_k, _c, h, w), _, _) = filter.info()?;

        let success = unsafe {
            cudnnSetConvolution2dDescriptor(
                conv_desc,
                ((h - 1) / 2) as c_int,
                ((w - 1) / 2) as c_int,
                1, 1, 1, 1,
                cudnnConvolutionMode_t::CrossCorrelation,
                cudnnDataType_t::Float
            )
        };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        let success = unsafe {
            cudnnSetConvolutionGroupCount(
                conv_desc,
                group_count as c_int
            )
        };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        let success = unsafe {
            cudnnSetConvolutionMathType(
                conv_desc,
                cudnnMathType_t::TensorOpMath
            )
        };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        Ok(Convolution(Unique::new(conv_desc).unwrap()))
    }

    pub fn compile_for(
        &self,
        handle: &Handle,
        input: &Tensor,
        filter: &Filter,
        output: &Tensor,
        algo: cudnnConvolutionFwdAlgo_t,
    ) -> Result<cudnnConvolutionFwdAlgoPerf_t, Error>
    {
        let mut fwd_algo = cudnnConvolutionFwdAlgoPerf_t {
            algo,
            status: 0,
            time: 0.0,
            memory: 0,
            determinism: cudnnDeterminism_t::NonDeterministic,
            math_type: cudnnMathType_t::DefaultMath,
            reserved: [0; 3]
        };

        let success = unsafe {
            cudnnGetConvolutionForwardWorkspaceSize(
                handle.as_ptr(),
                input.as_ptr(),
                filter.as_ptr(),
                self.as_ptr(),
                output.as_ptr(),
                algo,
                &mut fwd_algo.memory as *mut _ as *mut size_t
            )
        };

        if success != 0 {
            Err(Error::CudnnError(success))
        } else {
            Ok(fwd_algo)
        }
    }

    pub fn compile(
        &self,
        handle: &Handle,
        input: &Tensor,
        filter: &Filter,
        output: &Tensor,
        activation: &Activation
    ) -> Result<cudnnConvolutionFwdAlgoPerf_t, Error>
    {
        let (activation_mode, _relu_nan_opt, _coef) = activation.info()?;

        if activation_mode != cudnnActivationMode_t::ReLU {
            self.compile_for(
                handle,
                input,
                filter,
                output,
                cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm
            )
        } else {
            let mut count = 0;
            let mut fwd_algo = cudnnConvolutionFwdAlgoPerf_t {
                algo: cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
                status: 0,
                time: 0.0,
                memory: 0,
                determinism: cudnnDeterminism_t::NonDeterministic,
                math_type: cudnnMathType_t::TensorOpMathAllowConversion,
                reserved: [0; 3]
            };
            let success = unsafe {
                cudnnGetConvolutionForwardAlgorithm_v7(
                    handle.as_ptr(),
                    input.as_ptr(),
                    filter.as_ptr(),
                    self.as_ptr(),
                    output.as_ptr(),
                    1,
                    &mut count,
                    &mut fwd_algo
                )
            };

            if success != 0 || count == 0 {
                Err(Error::CudnnError(success))
            } else if fwd_algo.status != 0 {
                Err(Error::CudnnError(fwd_algo.status))
            } else {
                Ok(fwd_algo)
            }
        }
    }

    pub fn forward(
        &self,
        handle: &Handle,
        x: &Tensor,
        x_data: *const c_void,
        filter: &Filter,
        w_data: *const c_void,
        fwd_algo: &cudnnConvolutionFwdAlgoPerf_t,
        workspace: *mut c_void,
        bias: &Tensor,
        bias_data: *const c_void,
        activation: &Activation,
        y: &Tensor,
        y_data: *mut c_void
    ) -> Result<(), Error>
    {
        let success = unsafe {
            cudnnConvolutionBiasActivationForward(
                handle.as_ptr(),
                &ONE as *const _ as *const c_void,
                x.as_ptr(), x_data,
                filter.as_ptr(), w_data,
                self.as_ptr(),
                fwd_algo.algo, workspace, fwd_algo.memory,
                &ZERO as *const _ as *const c_void,
                y.as_ptr(), y_data,
                bias.as_ptr(), bias_data,
                activation.as_ptr(),
                y.as_ptr(), y_data
            )
        };

        if success != 0 {
            Err(Error::CudnnError(success))
        } else {
            Ok(())
        }
    }

    pub(super) fn as_ptr(&self) -> cudnnConvolutionDescriptor_t {
        self.0.as_ptr()
    }
}