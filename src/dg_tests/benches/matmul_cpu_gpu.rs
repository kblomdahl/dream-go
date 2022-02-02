// Copyright 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
extern crate test;
extern crate libc;

use test::Bencher;
use std::mem::size_of;

use dg_cuda as cuda;
use dg_cuda::cudnn;

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
/// * `reorder_type` - the filter & bias reorder type to set
///
unsafe fn gpu_matmul(
    bencher: &mut Bencher,
    batch_size: i32,
    num_inputs: i32,
    num_outputs: i32
) -> Result<(), cudnn::Status>
{
    let handle: cudnn::Handle = cudnn::Handle::new()?;
    let stream = cuda::Stream::default();
    let allocator = cuda::Native::default();
    let tensor_format = cudnn::TensorFormat::NCHW;
    let data_type = cudnn::DataType::Float;

    // the a input description that match the given configuration
    let in_desc = cudnn::TensorDescriptor::new(
        tensor_format,
        data_type,
        [batch_size, num_inputs, 1, 1]
    )?;
    let out_desc = cudnn::TensorDescriptor::new(
        tensor_format,
        data_type,
        [batch_size, num_outputs, 1, 1]
    )?;
    let skip_desc = cudnn::TensorDescriptor::new(
        tensor_format,
        data_type,
        [batch_size, num_outputs, 1, 1]
    )?;

    // the a bias description that match the given configuration
    let offset_desc = cudnn::TensorDescriptor::new(
        cudnn::TensorFormat::NCHW,
        data_type,
        [1, num_outputs, 1, 1]
    )?;

    // the a convolutional description that match the given configuration
    let conv_desc = cudnn::ConvolutionDescriptor::new(
        [0, 0],
        [1, 1],
        [1, 1],
        cudnn::ConvolutionMode::CrossCorrelation,
        data_type
    )?;

    // the relu descriptor
    let relu = cudnn::ActivationDescriptor::relu()?;

    // create a 3x3 filter description of the given data type
    let filter_desc = cudnn::FilterDescriptor::new(
        data_type,
        tensor_format,
        [num_outputs as i32, num_inputs as i32, 1, 1]
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
    let input_size = (batch_size * num_inputs) as usize;
    let mut input = cuda::malloc(size_of::<f32>() * input_size, &allocator).unwrap();
    input.copy_from_slice(&vec! [1.0; input_size], &stream).unwrap();

    // allocate the `weights` array and fill it with 0.1's.
    let weights_size = (num_inputs * num_outputs) as usize;
    let mut weights = cuda::malloc(size_of::<f32>() * weights_size, &allocator).unwrap();
    weights.copy_from_slice(&vec! [0.1; weights_size], &stream).unwrap();

    // allocate the `offset` array and initialize it with zeros.
    let offset_size = num_outputs as usize;
    let mut offset = cuda::malloc(size_of::<f32>() * offset_size, &allocator).unwrap();
    offset.copy_from_slice(&vec! [1.0; offset_size], &stream).unwrap();

    // allocate the `output` array, but leave it uninitialized since
    // we will never read into it until after the convolution.
    let output_size = (batch_size * num_outputs) as usize;
    let output = cuda::malloc(size_of::<f32>() * output_size, &allocator).unwrap();

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
        out_desc,
        skip_desc
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
        ).expect("convolution failed");
    });

    Ok(())
}

// -------- gpu  --------

#[bench]
fn gpu_08_2888x362(b: &mut Bencher) {
    unsafe { gpu_matmul(b, 8, 2888, 362).unwrap(); }
}

#[bench]
fn gpu_16_2888x362(b: &mut Bencher) {
    unsafe { gpu_matmul(b, 16, 2888, 362).unwrap(); }
}

#[bench]
fn gpu_08_722x1(b: &mut Bencher) {
    unsafe { gpu_matmul(b, 8, 722, 1).unwrap(); }
}

#[bench]
fn gpu_16_722x1(b: &mut Bencher) {
    unsafe { gpu_matmul(b, 16, 722, 1).unwrap(); }
}
