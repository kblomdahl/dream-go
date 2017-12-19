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
use dream_go::util::*;

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
    conv_algo: cudnn::ConvolutionFwdAlgo,
    conv_type: cudnn::DataType
)
    where f32: From<T>
{
    let mut handle: cudnn::Handle = ptr::null_mut();

    assert!(cudnn::cudnnCreate(&mut handle).is_ok());

    // the a input description that match the given configuration
    let mut inout_desc: cudnn::TensorDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateTensorDescriptor(&mut inout_desc).is_ok());
    assert!(cudnn::cudnnSetTensor4dDescriptor(
        inout_desc,
        tensor_format,
        data_type,
        1, num_features as i32, 19, 19
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
        assert!(cudnnSetConvolutionMathType(conv_desc, cudnn::MathType::TensorOpMath).is_ok());
    }

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
    let input_size = num_features * 361;

    assert!(cuda::cudaMalloc(&mut input, data_type.size() * input_size).is_ok());
    assert!(cuda::cudaMemcpy(
        input,
        vec! [T::from(1.0); input_size].as_ptr() as *const c_void,
        data_type.size() * input_size,
        cuda::MemcpyKind::HostToDevice
    ).is_ok());

    // allocate the `weights` array and initialize it with ones.
    let mut weights = ptr::null_mut();
    let weights_size = num_features * num_features * 3 * 3;

    assert!(cuda::cudaMalloc(&mut weights, data_type.size() * weights_size).is_ok());
    assert!(cuda::cudaMemcpy(
        weights,
        vec! [T::from(1.0); weights_size].as_ptr() as *const c_void,
        data_type.size() * weights_size,
        cuda::MemcpyKind::HostToDevice
    ).is_ok());

    // allocate the `output` array, but leave it uninitialized since
    // we will never read into it until after the convolution.
    let mut output = ptr::null_mut();
    let output_size = num_features * 361;

    assert!(cuda::cudaMalloc(&mut output, data_type.size() * output_size).is_ok());

    // run the benchmark
    bencher.iter(move || {
        let c_0: f32 = 0.0;
        let c_1: f32 = 1.0;

        assert!(cudnn::cudnnConvolutionForward(
            handle,
            &c_1,
            inout_desc, input,
            filter_desc, weights,
            conv_desc,
            conv_algo,
            workspace, workspace_size,
            &c_0,
            inout_desc, output
        ).is_ok());
    });

    // download the result from the GPU and check so that it has changed
    // from the value the host array was initialized with. We do thing
    // instead of a more complicated check due to the varying precision
    // of the different types.
    let not_zero = vec! [T::from(0.0); num_features * 361];

    assert!(cuda::cudaMemcpy(
        not_zero.as_ptr() as *mut c_void,
        output,
        data_type.size() * num_features * 361,
        cuda::MemcpyKind::DeviceToHost
    ).is_ok());

    assert!(cuda::cudaFree(input).is_ok());
    assert!(cuda::cudaFree(output).is_ok());
    assert!(cuda::cudaFree(weights).is_ok());

    for value in not_zero.iter() {
        let value = f32::from(value.clone());

        assert!(value != 0.0, "{}", value);
    }
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
    (version_major == 7 && version_minor == 0)
}

#[bench]
fn f32_256_winograd(b: &mut Bencher) {
    unsafe {
        bench_conv::<f32>(
            b,
            256,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Float,
            cudnn::ConvolutionFwdAlgo::Winograd,
            cudnn::DataType::Float
        );
    }
}

#[bench]
fn f16_256_winogradnonfused(b: &mut Bencher) {
    if !supports_f16() { return }

    unsafe {
        bench_conv::<f16::f16>(
            b,
            256,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::WinogradNonFused,
            cudnn::DataType::Half
        );
    }
}

#[bench]
fn f16_256_implicitprecompgemm(b: &mut Bencher) {
    if !supports_f16() { return }

    unsafe {
        bench_conv::<f16::f16>(
            b,
            256,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Half
        );
    }
}

#[bench]
fn p16_256_winograd(b: &mut Bencher) {
    unsafe {
        bench_conv::<f16::f16>(
            b,
            256,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::Winograd,
            cudnn::DataType::Float
        );
    }
}

#[bench]
fn p16_256_implicitprecompgemm(b: &mut Bencher) {
    unsafe {
        bench_conv::<f16::f16>(
            b,
            256,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Float
        );
    }
}

#[bench]
fn i8_256_implicitprecompgemm(b: &mut Bencher) {
    if !supports_i8() { return }

    unsafe {
        bench_conv::<q8::q8>(
            b,
            256,
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Int8,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Int32
        );
    }
}

#[bench]
fn i8x4_256_implicitprecompgemm(b: &mut Bencher) {
    if !supports_i8() { return }

    unsafe {
        bench_conv::<q8::q8>(
            b,
            256,
            cudnn::TensorFormat::NCHWVECTC,
            cudnn::DataType::Int8x4,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Int32
        );
    }
}

#[bench]
fn f32_128_winograd(b: &mut Bencher) {
    unsafe {
        bench_conv::<f32>(
            b,
            128,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Float,
            cudnn::ConvolutionFwdAlgo::Winograd,
            cudnn::DataType::Float
        );
    }
}

#[bench]
fn f16_128_winogradnonfused(b: &mut Bencher) {
    if !supports_f16() { return }

    unsafe {
        bench_conv::<f16::f16>(
            b,
            128,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::WinogradNonFused,
            cudnn::DataType::Half
        );
    }
}

#[bench]
fn f16_128_implicitprecompgemm(b: &mut Bencher) {
    if !supports_f16() { return }

    unsafe {
        bench_conv::<f16::f16>(
            b,
            128,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Half
        );
    }
}

#[bench]
fn p16_128_winograd(b: &mut Bencher) {
    unsafe {
        bench_conv::<f16::f16>(
            b,
            128,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::Winograd,
            cudnn::DataType::Float
        );
    }
}

#[bench]
fn p16_128_implicitprecompgemm(b: &mut Bencher) {
    unsafe {
        bench_conv::<f16::f16>(
            b,
            128,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Float
        );
    }
}

#[bench]
fn i8_128_implicitprecompgemm(b: &mut Bencher) {
    if !supports_i8() { return }

    unsafe {
        bench_conv::<q8::q8>(
            b,
            128,
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Int8,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Int32
        );
    }
}

#[bench]
fn i8x4_128_implicitprecompgemm(b: &mut Bencher) {
    if !supports_i8() { return }

    unsafe {
        bench_conv::<q8::q8>(
            b,
            128,
            cudnn::TensorFormat::NCHWVECTC,
            cudnn::DataType::Int8x4,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            cudnn::DataType::Int32
        );
    }
}
