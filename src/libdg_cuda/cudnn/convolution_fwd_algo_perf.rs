// Copyright 2020 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use crate::cudnn::*;

use libc::{c_float, c_int, size_t};

#[allow(non_camel_case_types)]
pub type cudnnConvolutionFwdAlgoPerf_t = ConvolutionFwdAlgoPerf;

#[link(name = "cudnn")]
extern {
    fn cudnnGetConvolutionForwardAlgorithm_v7(
        handle: cudnnHandle_t,
        x_desc: cudnnTensorDescriptor_t,
        w_desc: cudnnFilterDescriptor_t,
        conv_desc: cudnnConvolutionDescriptor_t,
        y_desc: cudnnTensorDescriptor_t,
        requested_algo_count: c_int,
        returned_algo_count: *mut c_int,
        perf_results: *mut cudnnConvolutionFwdAlgoPerf_t
    ) -> cudnnStatus_t;
}

#[repr(C)]
pub struct ConvolutionFwdAlgoPerf {
    algo: ConvolutionFwdAlgo,
    status: Status,
    time: c_float,
    memory: size_t,
    determinism: cudnnDeterminism_t,
    math_type: cudnnMathType_t,
    reserved: [c_int; 3]
}

impl ConvolutionFwdAlgoPerf {
    pub fn new(
        handle: &Handle,
        x: &TensorDescriptor,
        w: &FilterDescriptor,
        conv: &ConvolutionDescriptor,
        y: &TensorDescriptor,
    ) -> Result<Self, Status>
    {
        let mut count = 0;
        let mut out = Self {
            algo: ConvolutionFwdAlgo::ImplicitGemm,
            status: Status::Success,
            time: 0.0,
            memory: 0,
            determinism: Determinism::NonDeterministic,
            math_type: MathType::DefaultMath,
            reserved: [0; 3]
        };
        let status =
            unsafe {
                cudnnGetConvolutionForwardAlgorithm_v7(
                    **handle,
                    **x,
                    **w,
                    **conv,
                    **y,
                    1,
                    &mut count,
                    &mut out
                )
            };

        assert_eq!(count, 1);
        status.into_result(out)
    }

    pub fn algo(&self) -> ConvolutionFwdAlgo {
        self.algo
    }

    pub fn memory(&self) -> usize {
        self.memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_dilated_perf() {
        let handle = Handle::new().unwrap();
        let x = TensorDescriptor::new(
            TensorFormat::NHWC,
            DataType::Float,
            &[16, 256, 19, 19]
        ).unwrap();
        let w = FilterDescriptor::new(
            DataType::Float,
            TensorFormat::NHWC,
            &[256, 256, 3, 3]
        ).unwrap();
        let conv = ConvolutionDescriptor::new(
            &[2, 2],
            &[1, 1],
            &[2, 2],
            ConvolutionMode::CrossCorrelation,
            DataType::Float
        ).unwrap();
        let out = ConvolutionFwdAlgoPerf::new(
            &handle,
            &x,
            &w,
            &conv,
            &x
        );

        assert!(out.is_ok());

        let out = out.unwrap();
        assert_eq!(out.algo, ConvolutionFwdAlgo::ImplicitPrecompGemm);
        assert_eq!(out.math_type, MathType::DefaultMath);
    }
}