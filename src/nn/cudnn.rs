// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use libc::{c_double, c_int, c_void, size_t};

#[repr(i32)]
#[allow(dead_code)]
pub enum ActivationMode {
    Relu = 1,
    Tanh = 2
}

#[repr(i32)]
#[allow(dead_code)]
pub enum BatchNormMode {
    PerActivation = 0,
    Spatial = 1
}

#[repr(i32)]
#[allow(dead_code)]
pub enum ConvolutionMode {
    Convolution = 0,
    CrossCorrelation = 1
}

#[repr(i32)]
#[allow(dead_code)]
pub enum ConvolutionFwdAlgo {
    ImplicitPrecompGemm = 1,
    Winograd = 6,
}

#[repr(i32)]
#[allow(dead_code)]
pub enum DataType {
    Float = 0
}

#[repr(i32)]
#[derive(Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Status {
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 2,
    BadParam = 3,
    InternalError = 4,
    InvalidValue = 5,
    ArchMismatch = 6,
    MappingError = 7,
    ExecutionFailed = 8,
    NotSupported = 9,
    LicenseError = 10,
    RuntimePrerequisiteMissing = 11
}

#[repr(i32)]
#[allow(dead_code)]
pub enum NanPropagation {
    NotPropagateNan = 0,
    PropagateNan = 1
}

#[repr(i32)]
#[allow(dead_code)]
pub enum SoftmaxAlgorithm {
    Fast = 0,
    Accurate = 1,
    Log = 2
}

#[repr(i32)]
#[allow(dead_code)]
pub enum SoftmaxMode {
    Instance = 0,
    Channel = 1
}

#[repr(i32)]
#[allow(dead_code)]
pub enum TensorFormat {
    NCHW = 0
}

pub type ActivationDescriptor = *const c_void;
pub type ConvolutionDescriptor = *const c_void;
pub type FilterDescriptor = *const c_void;
pub type Handle = *const c_void;
pub type TensorDescriptor = *const c_void;

#[link(name = "cudnn")]
extern {
    pub fn cudnnCreate(handle: *mut Handle) -> Status;
    pub fn cudnnDestroy(handle: Handle) -> Status;

    pub fn cudnnCreateActivationDescriptor(activationDesc: *mut ActivationDescriptor) -> Status;
    pub fn cudnnDestroyActivationDescriptor(activationDesc: ActivationDescriptor) -> Status;
    pub fn cudnnSetActivationDescriptor(
        activationDesc: ActivationDescriptor,
        mode: ActivationMode,
        reluNanOpt: NanPropagation,
        coef: c_double
    ) -> Status;

    pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut TensorDescriptor) -> Status;
    pub fn cudnnDestroyTensorDescriptor(tensorDesc: TensorDescriptor) -> Status;
    pub fn cudnnSetTensor4dDescriptor(
        tensorDesc: TensorDescriptor,
        format: TensorFormat,
        dataType: DataType,
        n: c_int,  // batch
        c: c_int,  // channels
        h: c_int,  // height
        w: c_int,  // width
    ) -> Status;

    pub fn cudnnCreateFilterDescriptor(filterDesc: *mut FilterDescriptor) -> Status;
    pub fn cudnnDestroyFilterDescriptor(filterDesc: FilterDescriptor) -> Status;
    pub fn cudnnSetFilter4dDescriptor(
        filterDesc: FilterDescriptor,
        dataType: DataType,
        format: TensorFormat,
        k: c_int,  // output features
        c: c_int,  // input features
        h: c_int,  // height
        w: c_int,  // width
    ) -> Status;

    pub fn cudnnCreateConvolutionDescriptor(convDesc: *mut ConvolutionDescriptor) -> Status;
    pub fn cudnnDestroyConvolutionDescriptor(convDesc: ConvolutionDescriptor) -> Status;
    pub fn cudnnSetConvolution2dDescriptor(
        convDesc: ConvolutionDescriptor,
        pad_h: c_int,
        pad_w: c_int,
        u: c_int,
        v: c_int,
        dilation_h: c_int,
        dilation_w: c_int,
        mode: ConvolutionMode,
        computeType: DataType
    ) -> Status;

    pub fn cudnnAddTensor(
        handle: Handle,
        alpha: *const f32,
        aDesc: TensorDescriptor,
        A: *const c_void,
        beta: *const f32,
        cDesc: TensorDescriptor,
        C: *mut c_void
    ) -> Status;

    pub fn cudnnActivationForward(
        handle: Handle,
        activationDesc: ActivationDescriptor,
        alpha: *const f32,
        srcDesc: TensorDescriptor,
        src: *const c_void,
        beta: *const f32,
        destDesc: TensorDescriptor,
        dest: *mut c_void,
    ) -> Status;

    pub fn cudnnBatchNormalizationForwardInference(
        handle: Handle,
        mode: BatchNormMode,
        alpha: *const f32,
        beta: *const f32,
        xDesc: TensorDescriptor,
        x: *const c_void,
        yDesc: TensorDescriptor,
        y: *mut c_void,
        bnScaleBiasMeanVarDesc: TensorDescriptor,
        bnScale: *const c_void,
        bnBias: *const c_void,
        estimatedMean: *const c_void,
        estimatedVariance: *const c_void,
        epsilon: c_double
    ) -> Status;

    pub fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: Handle,
        xDesc: TensorDescriptor,
        wDesc: FilterDescriptor,
        convDesc: ConvolutionDescriptor,
        yDesc: TensorDescriptor,
        algo: ConvolutionFwdAlgo,
        sizeInBytes: *mut size_t
    ) -> Status;

    pub fn cudnnConvolutionForward(
        handle: Handle,
        alpha: *const f32,
        xDesc: TensorDescriptor,
        x: *const c_void,
        wDesc: FilterDescriptor,
        w: *const c_void,
        convDesc: ConvolutionDescriptor,
        algo: ConvolutionFwdAlgo,
        workSpace: *mut c_void,
        workSpaceSizeInBytes: size_t,
        beta: *const f32,
        yDesc: TensorDescriptor,
        y: *mut c_void
    ) -> Status;

    pub fn cudnnSoftmaxForward(
        handle: Handle,
        algorithm: SoftmaxAlgorithm,
        mode: SoftmaxMode,
        alpha: *const f32,
        xDesc: TensorDescriptor,
        x: *const c_void,
        beta: *const f32,
        yDesc: TensorDescriptor,
        y: *mut c_void
    ) -> Status;
}
