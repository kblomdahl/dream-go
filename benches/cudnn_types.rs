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
//
#![feature(test)]

extern crate dream_go;
extern crate test;
extern crate libc;

use std::ptr;
use test::Bencher;
use libc::{c_void};

use dream_go::nn::ffi::*;
use dream_go::util::types::*;

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
struct q8 { value: i8 }

impl From<f32> for q8 {
    fn from(other: f32) -> q8 {
        q8 { value: (127.0 * other).round() as i8 }
    }
}

impl From<q8> for f32 {
    fn from(other: q8) -> f32 {
        other.value as f32 / 127.0
    }
}

/// Benchmark the given cuDNN configuration using a single convolutional
/// layer.
///
/// # Arguments
///
/// * `bencher` - the benchmarker
/// * `num_features` - the number of features
/// * `tensor_format` - the format of the input / output arrays
/// * `data_type` - the data type of the input / output / weight arrays
/// * `conv_algo` - the convolutional algorithm to use
/// * `conv_type` - the data type to use internally in the convolution
///
unsafe fn bench_conv<T: From<f32> + Clone>(
    bencher: &mut Bencher,
    num_features: usize,
    tensor_format: cudnn::TensorFormat,
    data_type: cudnn::DataType,
    offset_type: cudnn::DataType,
    conv_algo: cudnn::ConvolutionFwdAlgo,
    conv_type: cudnn::DataType
)
    where f32: From<T>
{
    let mut handle: cudnn::Handle = ptr::null_mut();
    const BATCH_SIZE: usize = 16;

    assert!(cudnn::cudnnCreate(&mut handle).is_ok());

    // the a input description that match the given configuration
    let mut inout_desc: cudnn::TensorDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateTensorDescriptor(&mut inout_desc).is_ok());
    assert!(cudnn::cudnnSetTensor4dDescriptor(
        inout_desc,
        tensor_format,
        data_type,
        BATCH_SIZE as i32, num_features as i32, 19, 19
    ).is_ok());

    // the a bias description that match the given configuration
    let mut offset_desc: cudnn::TensorDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateTensorDescriptor(&mut offset_desc).is_ok());
    assert!(cudnn::cudnnSetTensor4dDescriptor(
        offset_desc,
        if tensor_format == cudnn::TensorFormat::NCHWVECTC { cudnn::TensorFormat::NCHW } else { tensor_format },
        offset_type,
        1, num_features as i32, 1, 1
    ).is_ok());

    // the a convolutional description that match the given configuration
    let mut conv_desc: cudnn::ConvolutionDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateConvolutionDescriptor(&mut conv_desc).is_ok());
    assert!(cudnn::cudnnSetConvolution2dDescriptor(
        conv_desc,
        1, 1, 1, 1, 1, 1,
        cudnn::ConvolutionMode::CrossCorrelation,
        conv_type
    ).is_ok());

    #[cfg(feature = "tensor-core")]
    {
        assert!(cudnn::cudnnSetConvolutionMathType(conv_desc, cudnn::MathType::TensorOpMath).is_ok());
    }

    // the relu descriptor
    let mut relu: cudnn::ActivationDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateActivationDescriptor(&mut relu).is_ok());
    assert!(cudnn::cudnnSetActivationDescriptor(
        relu,
        cudnn::ActivationMode::Relu,
        cudnn::NanPropagation::NotPropagateNan,
        0.0
    ).is_ok());

    // create a 3x3 filter description of the given data type
    let mut filter_desc: cudnn::FilterDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateFilterDescriptor(&mut filter_desc).is_ok());
    assert!(cudnn::cudnnSetFilter4dDescriptor(
        filter_desc,
        data_type,
        tensor_format,
        num_features as i32, num_features as i32, 3, 3
    ).is_ok());

    // figure out how large of a workspace we need given the convolution
    // configuration.
    let mut workspace_size: usize = 0;
    let mut workspace = ptr::null_mut();

    assert!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        inout_desc,
        filter_desc,
        conv_desc,
        inout_desc,
        conv_algo,
        &mut workspace_size
    ).is_ok());

    assert!(cuda::cudaMalloc(&mut workspace, workspace_size).is_ok());

    // allocate the `input` array and initialize it with ones.
    let mut input = ptr::null_mut();
    let input_size = BATCH_SIZE * num_features * 361;

    assert!(cuda::cudaMalloc(&mut input, data_type.size() * input_size).is_ok());
    assert!(cuda::cudaMemcpy(
        input,
        vec! [T::from(1.0); input_size].as_ptr() as *const c_void,
        data_type.size() * input_size,
        cuda::MemcpyKind::HostToDevice
    ).is_ok());

    // allocate the `weights` array and fill it with 0.1's.
    let mut weights = ptr::null_mut();
    let weights_size = num_features * num_features * 3 * 3;

    assert!(cuda::cudaMalloc(&mut weights, data_type.size() * weights_size).is_ok());
    assert!(cuda::cudaMemcpy(
        weights,
        vec! [T::from(0.1); weights_size].as_ptr() as *const c_void,
        data_type.size() * weights_size,
        cuda::MemcpyKind::HostToDevice
    ).is_ok());

    // allocate the `offset` array and initialize it with zeros.
    let mut offset = ptr::null_mut();
    let offset_size = num_features;

    assert!(cuda::cudaMalloc(&mut offset, offset_type.size() * offset_size).is_ok());
    assert!(cuda::cudaMemcpy(
        offset,
        vec! [T::from(0.0); offset_size].as_ptr() as *const c_void,
        offset_type.size() * offset_size,
        cuda::MemcpyKind::HostToDevice
    ).is_ok());

    // allocate the `output` array, but leave it uninitialized since
    // we will never read into it until after the convolution.
    let mut output = ptr::null_mut();
    let output_size = BATCH_SIZE * num_features * 361;

    assert!(cuda::cudaMalloc(&mut output, data_type.size() * output_size).is_ok());

    // run the benchmark
    bencher.iter(move || {
        let c_0: f32 = 0.0;
        let c_1: f32 = 1.0;

        assert!(cudnn::cudnnConvolutionBiasActivationForward(
            handle,
            &c_1,
            inout_desc, input,
            filter_desc, weights,
            conv_desc,
            conv_algo,
            workspace, workspace_size,
            &c_0,
            inout_desc, output,
            offset_desc, offset,
            relu,
            inout_desc, output
        ).is_ok());
    });

    assert!(cuda::cudaFree(input).is_ok());
    assert!(cuda::cudaFree(output).is_ok());
    assert!(cuda::cudaFree(offset).is_ok());
    assert!(cuda::cudaFree(weights).is_ok());
}

/// Returns true if the current device supports `i8`.
fn supports_i8() -> bool {
    let mut version_major: i32 = 0;
    let mut version_minor: i32 = 0;

    unsafe {
        assert!(cuda::cudaDeviceGetAttribute(&mut version_major, cuda::DeviceAttr::ComputeCapabilityMajor, 0).is_ok());
        assert!(cuda::cudaDeviceGetAttribute(&mut version_minor, cuda::DeviceAttr::ComputeCapabilityMinor, 0).is_ok());
    }

    (version_major == 6 && version_minor >= 1) ||
    (version_major >= 7 && version_minor >= 0)
}

/// Returns true if the current device supports `f16` (in a
/// sensible way).
fn supports_f16() -> bool {
    let mut version_major: i32 = 0;
    let mut version_minor: i32 = 0;

    unsafe {
        assert!(cuda::cudaDeviceGetAttribute(&mut version_major, cuda::DeviceAttr::ComputeCapabilityMajor, 0).is_ok());
        assert!(cuda::cudaDeviceGetAttribute(&mut version_minor, cuda::DeviceAttr::ComputeCapabilityMinor, 0).is_ok());
    }

    (version_major == 6 && version_minor == 0) ||
    (version_major == 6 && version_minor == 2) ||
    (version_major >= 7)
}

// -------- 32-bit floating point --------

#[bench]
fn f32_128_winograd(b: &mut Bencher) {
    unsafe {
        bench_conv::<f32>(
            b,
            128,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Float, cudnn::DataType::Float,
            cudnn::ConvolutionFwdAlgo::Winograd,
            cudnn::DataType::Float
        );
    }
}

// -------- 16-bit floating point --------

#[bench]
fn f16_128_winogradnonfused(b: &mut Bencher) {
    if !supports_f16() { return }

    unsafe {
        bench_conv::<f16>(
            b,
            128,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half, cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::WinogradNonFused,
            cudnn::DataType::Half
        );
    }
}

#[bench]
fn f16_128_implicitprecompgemm_nhwc(b: &mut Bencher) {
    if !supports_f16() { return }

    unsafe {
        bench_conv::<f16>(
            b,
            128,
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Half, cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Half
        );
    }
}

#[bench]
fn f16_128_implicitprecompgemm(b: &mut Bencher) {
    if !supports_f16() { return }

    unsafe {
        bench_conv::<f16>(
            b,
            128,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half, cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Half
        );
    }
}

// -------- 16-bit pseudo floating point --------

#[bench]
fn p16_128_winograd(b: &mut Bencher) {
    unsafe {
        bench_conv::<f16>(
            b,
            128,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half, cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::Winograd,
            cudnn::DataType::Float
        );
    }
}

#[bench]
fn p16_128_winogradnonfused(b: &mut Bencher) {
    unsafe {
        bench_conv::<f16>(
            b,
            128,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half, cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::WinogradNonFused,
            cudnn::DataType::Float
        );
    }
}

#[bench]
fn p16_128_implicitprecompgemm(b: &mut Bencher) {
    unsafe {
        bench_conv::<f16>(
            b,
            128,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half, cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Float
        );
    }
}

#[bench]
fn p16_128_implicitprecompgemm_nhwc(b: &mut Bencher) {
    unsafe {
        bench_conv::<f16>(
            b,
            128,
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Half, cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Float
        );
    }
}

// -------- 8-bit signed integer --------

#[bench]
fn i8_128_implicitprecompgemm(b: &mut Bencher) {
    if !supports_i8() { return }

    unsafe {
        bench_conv::<q8>(
            b,
            128,
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Int8, cudnn::DataType::Float,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Int32
        );
    }
}

#[bench]
fn i8x4_128_implicitprecompgemm(b: &mut Bencher) {
    if !supports_i8() { return }

    unsafe {
        bench_conv::<q8>(
            b,
            128,
            cudnn::TensorFormat::NCHWVECTC,
            cudnn::DataType::Int8x4, cudnn::DataType::Float,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Int32
        );
    }
}
