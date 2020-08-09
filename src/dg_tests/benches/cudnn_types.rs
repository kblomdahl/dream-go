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
#![feature(test)]

extern crate dg_cuda;
extern crate dg_utils;
extern crate dg_nn;
extern crate test;
extern crate libc;

use test::Bencher;
use std::mem::size_of;

use dg_cuda as cuda;
use dg_cuda::cudnn;
use dg_utils::types::*;

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
    conv_type: cudnn::DataType
) -> Result<(), cudnn::Status>
    where f32: From<T>
{
    let handle: cudnn::Handle = cudnn::Handle::new()?;
    let stream = cuda::Stream::default();
    let allocator = cuda::Native::default();
    const BATCH_SIZE: usize = 16;

    // the a input description that match the given configuration
    let in_desc = cudnn::TensorDescriptor::new(
        tensor_format,
        data_type,
        &[BATCH_SIZE as i32, num_features as i32, 19, 19]
    )?;
    let out_desc = cudnn::TensorDescriptor::new(
        tensor_format,
        data_type,
        &[BATCH_SIZE as i32, num_features as i32, 19, 19]
    )?;

    // the a bias description that match the given configuration
    let offset_desc = cudnn::TensorDescriptor::new(
        if tensor_format == cudnn::TensorFormat::NCHW_VECT_C { cudnn::TensorFormat::NCHW } else { tensor_format },
        offset_type,
        &[1, num_features as i32, 1, 1]
    )?;

    // the a convolutional description that match the given configuration
    let conv_desc = cudnn::ConvolutionDescriptor::new(
        &[1, 1],
        &[1, 1],
        &[1, 1],
        cudnn::ConvolutionMode::CrossCorrelation,
        conv_type
    )?;

    // the relu descriptor
    let relu = cudnn::ActivationDescriptor::relu()?;

    // create a 3x3 filter description of the given data type
    let filter_desc = cudnn::FilterDescriptor::new(
        data_type,
        tensor_format,
        &[num_features as i32, num_features as i32, 3, 3]
    )?;

    // figure out how large of a workspace we need given the convolution
    // configuration.
    let fwd_algo_perf = cudnn::ConvolutionFwdAlgoPerf::new(
        &handle,
        &in_desc,
        &filter_desc,
        &conv_desc,
        &out_desc
    )?;
    let workspace = cuda::malloc(fwd_algo_perf.memory(), &allocator).unwrap();

    // allocate the `input` array and initialize it with ones.
    let input_size = BATCH_SIZE * num_features * 19 * 19;
    let mut input = cuda::malloc(size_of::<T>() * input_size, &allocator).unwrap();
    input.copy_from_slice(&vec! [T::from(1.0); input_size], &stream).unwrap();

    // allocate the `weights` array and fill it with 0.1's.
    let weights_size = num_features * num_features * 3 * 3;
    let mut weights = cuda::malloc(size_of::<T>() * weights_size, &allocator).unwrap();
    weights.copy_from_slice(&vec! [T::from(0.1); weights_size], &stream).unwrap();

    // allocate the `offset` array and initialize it with zeros.
    let offset_size = num_features;
    let mut offset = cuda::malloc(size_of::<T>() * offset_size, &allocator).unwrap();
    offset.copy_from_slice(&vec! [T::from(0.0); offset_size], &stream).unwrap();

    // allocate the `output` array, but leave it uninitialized since
    // we will never read into it until after the convolution.
    let output_size = BATCH_SIZE * num_features * 361;
    let output = cuda::malloc(size_of::<T>() * output_size, &allocator).unwrap();

    // run the benchmark
    let conv_bias_act = cudnn::ConvolutionBiasActivation::new(
        &handle,
        1.0,
        in_desc,
        filter_desc,
        conv_desc,
        0.0,
        offset_desc,
        relu,
        out_desc
    )?;

    bencher.iter(move || {
        conv_bias_act.forward(
            &handle,
            input.as_ptr(),
            weights.as_ptr(),
            workspace.as_ptr(),
            fwd_algo_perf.memory(),
            output.as_ptr(),
            offset.as_ptr(),
            output.as_ptr()
        ).unwrap();
    });

    Ok(())
}

/// The number of channels to use in *all* benchmarks.
const NUM_FEATURES: usize = 192;

/// Returns true if the current device supports `i8`.
fn supports_i8() -> bool {
    let (version_major, version_minor) = cuda::Device::default().compute_capability().unwrap();

    (version_major == 6 && version_minor >= 1) ||
    (version_major >= 7 && version_minor >= 0)
}

/// Returns true if the current device supports `f16` (in a
/// sensible way).
fn supports_f16() -> bool {
    let (version_major, version_minor) = cuda::Device::default().compute_capability().unwrap();

    (version_major == 6 && version_minor == 0) ||
    (version_major == 6 && version_minor == 2) ||
    (version_major >= 7)
}

// -------- 32-bit floating point --------

#[bench]
fn f32_compute_type_f32(b: &mut Bencher) {
    unsafe {
        bench_conv::<f32>(
            b,
            NUM_FEATURES,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Float, cudnn::DataType::Float,
            cudnn::DataType::Float
        ).unwrap();
    }
}

// -------- 16-bit floating point --------

#[bench]
fn f16_nchw_compute_type_f32(b: &mut Bencher) {
    if !supports_f16() { return }

    unsafe {
        bench_conv::<f16>(
            b,
            NUM_FEATURES,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half, cudnn::DataType::Half,
            cudnn::DataType::Float
        ).unwrap();
    }
}

#[bench]
fn f16_nhwc_compute_type_f32(b: &mut Bencher) {
    if !supports_f16() { return }

    unsafe {
        bench_conv::<f16>(
            b,
            NUM_FEATURES,
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Half, cudnn::DataType::Half,
            cudnn::DataType::Float
        ).unwrap();
    }
}

#[bench]
fn f16_nchw_compute_type_f16(b: &mut Bencher) {
    if !supports_f16() { return }

    unsafe {
        bench_conv::<f16>(
            b,
            NUM_FEATURES,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half, cudnn::DataType::Half,
            cudnn::DataType::Half
        ).unwrap();
    }
}

#[bench]
fn f16_nhwc_compute_type_f16(b: &mut Bencher) {
    if !supports_f16() { return }

    unsafe {
        bench_conv::<f16>(
            b,
            NUM_FEATURES,
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Half, cudnn::DataType::Half,
            cudnn::DataType::Half
        ).unwrap();
    }
}

// -------- 8-bit signed integer --------

#[bench]
fn i8_nhwc(b: &mut Bencher) {
    if !supports_i8() { return }

    unsafe {
        bench_conv::<q8>(
            b,
            NUM_FEATURES,
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Int8, cudnn::DataType::Float,
            cudnn::DataType::Int32
        ).unwrap();
    }
}

#[bench]
fn i8x4_nhwcvectc(b: &mut Bencher) {
    if !supports_i8() { return }

    unsafe {
        bench_conv::<q8>(
            b,
            NUM_FEATURES,
            cudnn::TensorFormat::NCHW_VECT_C,
            cudnn::DataType::Int8x4, cudnn::DataType::Float,
            cudnn::DataType::Int32
        ).unwrap();
    }
}

#[bench]
fn i8x32_nhwcvectc(b: &mut Bencher) {
    if !supports_i8() { return }

    unsafe {
        bench_conv::<q8>(
            b,
            NUM_FEATURES,
            cudnn::TensorFormat::NCHW_VECT_C,
            cudnn::DataType::Int8x32, cudnn::DataType::Float,
            cudnn::DataType::Int32
        ).unwrap();
    }
}