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

use libc::{c_void, c_int, c_float, size_t};

#[repr(i32)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)] pub enum cudnnDataType_t {
    Float = 0,
    Half = 2
}

impl cudnnDataType_t {
    pub fn size_in_bytes(self) -> usize {
        match self {
            cudnnDataType_t::Float => 4,
            cudnnDataType_t::Half => 2
        }
    }
}

#[repr(i32)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)] pub enum cudnnTensorFormat_t {
    NCHW = 0,
    NHWC = 1
}

#[repr(i32)]
#[allow(non_camel_case_types)] pub enum cudnnConvolutionMode_t {
    CrossCorrelation = 1
}

#[repr(i32)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)] pub enum cudnnActivationMode_t {
    Sigmoid = 0,
    ReLU = 1,
    Tanh = 2,
    Identity = 5
}

#[repr(i32)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)] pub enum cudnnNanPropagation_t {
    NotPropagateNaN = 0
}

#[repr(i32)]
#[derive(Debug)]
#[allow(non_camel_case_types)] pub enum cudnnDeterminism_t {
    NonDeterministic = 0,
    Deterministic = 1
}

#[repr(i32)]
#[derive(Debug)]
#[allow(non_camel_case_types)] pub enum cudnnConvolutionFwdPreference_t {
    NoWorkspace = 0,
    PreferFastest = 1,
    SpecifyWorkspaceLimit = 2
}

#[repr(i32)]
#[derive(Debug)]
#[allow(non_camel_case_types)] pub enum cudnnMathType_t {
    DefaultMath = 0,
    TensorOpMath = 1,
    TensorOpMathAllowConversion = 2
}

#[repr(i32)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)] pub enum cudnnConvolutionFwdAlgo_t {
    ImplicitGemm = 0,
    ImplicitPrecompGemm = 1,
    Gemm = 2,
    Direct = 3,
    FFT = 4,
    FFTTiling = 5,
    Winograd = 6,
    WinogradNonFused = 7
}

#[repr(i32)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)] pub enum cudnnPoolingMode_t {
    Max = 0,
    AvgCountIncludePadding = 1,
    AvgCountExcludePadding = 2,
    MaxDeterministic = 3
}

#[repr(i32)]
#[allow(non_camel_case_types)] pub enum cudnnSoftmaxAlgorithm_t {
    Fast = 0,
    Accurate = 1,
    Log = 2
}

#[repr(i32)]
#[allow(non_camel_case_types)] pub enum cudnnSoftmaxMode_t {
    Instance = 0,
    Channel = 1
}

#[repr(i32)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)] pub enum cudnnOpTensorOp_t {
    Add = 0,
    Mul = 1,
    Min = 2,
    Max = 3,
    Sqrt = 4,
    Not = 5
}

#[repr(C)]
#[derive(Debug)]
#[allow(non_camel_case_types)] pub struct cudnnConvolutionFwdAlgoPerf_t {
    pub algo: cudnnConvolutionFwdAlgo_t,
    pub status: c_int,
    pub time: c_float,
    pub memory: size_t,
    pub determinism: cudnnDeterminism_t,
    pub math_type: cudnnMathType_t,
    pub reserved: [c_int; 3]
}

#[allow(non_camel_case_types)] pub type cudnnHandle_t = *mut c_void;
#[allow(non_camel_case_types)] pub type cudnnTensorDescriptor_t = *mut c_void;
#[allow(non_camel_case_types)] pub type cudnnPoolingDescriptor_t = *mut c_void;
#[allow(non_camel_case_types)] pub type cudnnFilterDescriptor_t = *mut c_void;
#[allow(non_camel_case_types)] pub type cudnnConvolutionDescriptor_t = *mut c_void;
#[allow(non_camel_case_types)] pub type cudnnActivationDescriptor_t = *mut c_void;
#[allow(non_camel_case_types)] pub type cudnnOpTensorDescriptor_t = *mut c_void;

mod handle;
mod activation;
mod filter;
mod tensor;
mod convolution;
mod pooling;
mod scale;
mod softmax;
mod op_tensor;

pub use self::handle::Handle;
pub use self::activation::Activation;
pub use self::convolution::Convolution;
pub use self::pooling::Pooling;
pub use self::scale::Scale;
pub use self::softmax::Softmax;
pub use self::filter::Filter;
pub use self::tensor::Tensor;
pub use self::op_tensor::OpTensor;
