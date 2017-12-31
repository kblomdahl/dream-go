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
extern crate dream_go;
extern crate libc;
extern crate ordered_float;

use libc::c_void;
use ordered_float::*;
use std::ptr;

use dream_go::nn::ffi::cuda;
use dream_go::nn::ffi::cudnn;
use dream_go::util::q8::*;

const FILTER_SCALE: f32 = 0.09;  // 0.09;
const INPUT_SCALE: f32 = 0.4;  // 0.4;
const OUTPUT_SCALE: f32 = 0.81;  // 0.81;

/// Create a filter filled with 4x4 planes, where each output plane
/// is one of the following
/// 
///      ( 0.01  0.01  0.01 )
/// 4 x  ( 0.01  0.01  0.01 )
///      ( 0.01  0.01  0.01 )
/// 
///      ( 0.03  0.03  0.03 )
/// 4 x  ( 0.03  0.03  0.03 )
///      ( 0.03  0.03  0.03 )
/// 
///      ( 0.06  0.06  0.06 )
/// 4 x  ( 0.06  0.06  0.06 )
///      ( 0.06  0.06  0.06 )
/// 
///      ( 0.09  0.09  0.09 )
/// 4 x  ( 0.09  0.09  0.09 )
///      ( 0.09  0.09  0.09 )
/// 
unsafe fn create_filter() -> (cudnn::FilterDescriptor, *const c_void) {
    let filter_planes = vec! [
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,

        0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
        0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
        0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
        0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,

        0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
        0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
        0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
        0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,

        0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09,
        0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09,
        0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09,
        0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09,
    ];
    let filter_total = filter_planes.iter()
        .map(|&v| q8::unscaled(v / FILTER_SCALE))  // explode the values into the full [-1,+1] range
        .collect::<Vec<q8>>();

    // copy the filter data to the GPU
    let mut filter_data: *mut c_void = ptr::null_mut();

    assert!(cuda::cudaMalloc(&mut filter_data, 16 * 9).is_ok());
    assert!(cuda::cudaMemcpy(
        filter_data,
        filter_total.as_ptr() as *const c_void,
        16 * 9,
        cuda::MemcpyKind::HostToDevice
    ).is_ok());

    // create the cuDNN descriptor for the filter
    let mut filter_desc: cudnn::FilterDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateFilterDescriptor(&mut filter_desc).is_ok());
    assert!(cudnn::cudnnSetFilter4dDescriptor(
        filter_desc,
        cudnn::DataType::Int8,
        cudnn::TensorFormat::NHWC,
        4, 4, 3, 3
    ).is_ok());

    (filter_desc, filter_data)
}

/// Create an offset tensor with the following values:
/// 
///   0.2  0.1  0.0  -0.1
unsafe fn create_offset() -> (cudnn::TensorDescriptor, *const c_void) {
    let output_total = vec! [
        127.0 * 0.2f32 / OUTPUT_SCALE,
        127.0 * 0.1f32 / OUTPUT_SCALE,
        127.0 * 0.0f32 / OUTPUT_SCALE,
        127.0 *-0.1f32 / OUTPUT_SCALE
    ];

    // copy the tensor to the GPU
    let mut offset_data: *mut c_void = ptr::null_mut();

    assert!(cuda::cudaMalloc(&mut offset_data, 4 * 4).is_ok());
    assert!(cuda::cudaMemcpy(
        offset_data,
        output_total.as_ptr() as *const c_void,
        4 * 4,
        cuda::MemcpyKind::HostToDevice
    ).is_ok());

    // create the cuDNN descriptor for the input tensor
    let mut offset_desc: cudnn::TensorDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateTensorDescriptor(&mut offset_desc).is_ok());
    assert!(cudnn::cudnnSetTensor4dDescriptor(
        offset_desc,
        cudnn::TensorFormat::NHWC,
        cudnn::DataType::Float,
        1, 4, 1, 1
    ).is_ok());

    (offset_desc, offset_data)
}

/// Create an input tensor with 4 planes, where each plane is a 5x5
/// matrix with the following values:
/// 
///   0.1  0.1  0.1  0.1  0.1
///   0.1  0.1  0.1  0.1  0.1
///   0.1  0.1  0.1  0.1  0.1
///   0.1  0.1  0.1  0.1  0.1
///   0.1  0.1  0.1  0.1  0.1
/// 
///   0.2  0.2  0.2  0.2  0.2
///   0.2  0.2  0.2  0.2  0.2
///   0.2  0.2  0.2  0.2  0.2
///   0.2  0.2  0.2  0.2  0.2
///   0.2  0.2  0.2  0.2  0.2
///
///   0.3  0.3  0.3  0.3  0.3
///   0.3  0.3  0.3  0.3  0.3
///   0.3  0.3  0.3  0.3  0.3
///   0.3  0.3  0.3  0.3  0.3
///   0.3  0.3  0.3  0.3  0.3
/// 
///   0.4  0.4  0.4  0.4  0.4
///   0.4  0.4  0.4  0.4  0.4
///   0.4  0.4  0.4  0.4  0.4
///   0.4  0.4  0.4  0.4  0.4
/// 
unsafe fn create_input() -> (cudnn::TensorDescriptor, *const c_void) {
    let plane_1 = vec! [0.1].into_iter().cycle().take(25).collect::<Vec<f32>>();
    let plane_2 = vec! [0.2].into_iter().cycle().take(25).collect::<Vec<f32>>();
    let plane_3 = vec! [0.3].into_iter().cycle().take(25).collect::<Vec<f32>>();
    let plane_4 = vec! [0.4].into_iter().cycle().take(25).collect::<Vec<f32>>();

    // write the tensor in NHWC format, 
    let mut input_total = Vec::with_capacity(4*25);

    for i in 0..25 {
        input_total.push(q8::unscaled(plane_1[i] / INPUT_SCALE));
        input_total.push(q8::unscaled(plane_2[i] / INPUT_SCALE));
        input_total.push(q8::unscaled(plane_3[i] / INPUT_SCALE));
        input_total.push(q8::unscaled(plane_4[i] / INPUT_SCALE));
    }

    // copy the tensor to the GPU
    let mut input_data: *mut c_void = ptr::null_mut();

    assert!(cuda::cudaMalloc(&mut input_data, 4 * 25).is_ok());
    assert!(cuda::cudaMemcpy(
        input_data,
        input_total.as_ptr() as *const c_void,
        4 * 25,
        cuda::MemcpyKind::HostToDevice
    ).is_ok());

    // create the cuDNN descriptor for the input tensor
    let mut input_desc: cudnn::TensorDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateTensorDescriptor(&mut input_desc).is_ok());
    assert!(cudnn::cudnnSetTensor4dDescriptor(
        input_desc,
        cudnn::TensorFormat::NHWC,
        cudnn::DataType::Int8,
        1, 4, 5, 5
    ).is_ok());

    (input_desc, input_data)
}

/// Create an empty output tensor with 4 planes.
unsafe fn create_output() -> (cudnn::TensorDescriptor, *mut c_void) {
    // copy the tensor to the GPU
    let mut output_data: *mut c_void = ptr::null_mut();

    assert!(cuda::cudaMalloc(&mut output_data, 4 * 25).is_ok());

    // create the cuDNN descriptor for the input tensor
    let mut output_desc: cudnn::TensorDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateTensorDescriptor(&mut output_desc).is_ok());
    assert!(cudnn::cudnnSetTensor4dDescriptor(
        output_desc,
        cudnn::TensorFormat::NHWC,
        cudnn::DataType::Int8,
        1, 4, 5, 5
    ).is_ok());

    (output_desc, output_data)
}

unsafe fn create_cross_correlation(
    handle: cudnn::Handle,
    input: cudnn::TensorDescriptor,
    output: cudnn::TensorDescriptor,
    weights: cudnn::FilterDescriptor
) -> (cudnn::ConvolutionDescriptor, *mut c_void, usize)
{
    // create the convolution descriptor
    let mut conv_desc: cudnn::ConvolutionDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateConvolutionDescriptor(&mut conv_desc).is_ok());
    assert!(cudnn::cudnnSetConvolution2dDescriptor(
        conv_desc,
        1, 1,
        1, 1,
        1, 1,
        cudnn::ConvolutionMode::CrossCorrelation,
        cudnn::DataType::Int32
    ).is_ok());

    // allocate the necessary workspace
    let mut workspace_size: usize = 0;
    let mut workspace_data: *mut c_void = ptr::null_mut();

    assert!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        input,
        weights,
        conv_desc,
        output,
        cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
        &mut workspace_size
    ).is_ok());

    assert!(cuda::cudaMalloc(&mut workspace_data, workspace_size).is_ok());

    (conv_desc, workspace_data, workspace_size)
}

unsafe fn create_relu() -> cudnn::ActivationDescriptor {
    let mut relu_desc: cudnn::ActivationDescriptor = ptr::null_mut();

    assert!(cudnn::cudnnCreateActivationDescriptor(&mut relu_desc).is_ok());
    assert!(cudnn::cudnnSetActivationDescriptor(
        relu_desc,
        cudnn::ActivationMode::Relu,
        cudnn::NanPropagation::NotPropagateNan,
        0.0
    ).is_ok());

    relu_desc
}

/// Check that 2 * 0.1 = 0.2
unsafe fn __2x0_1() {
    let mut handle: cudnn::Handle = ptr::null_mut();

    cudnn::cudnnCreate(&mut handle);

    // create the I/O, weights, and convolution
    let (filter_desc, filter_data) = create_filter();
    let (input_desc, input_data) = create_input();
    let (offset_desc, offset_data) = create_offset();
    let (output_desc, output_data) = create_output();
    let (conv_desc, workspace_data, workspace_size) = create_cross_correlation(handle, input_desc, output_desc, filter_desc);
    let relu_desc = create_relu();

    // perform the convolution
    let mut input = vec! [0i8; 4 * 25];
    let mut filter = vec! [0i8; 16 * 9];
    let mut output = vec! [0i8; 4 * 25];
    let (c_0, c_1) = (0.0f32, (FILTER_SCALE * INPUT_SCALE) / (OUTPUT_SCALE * 127.0));

    assert!(cudnn::cudnnConvolutionBiasActivationForward(
        handle,
        &c_1,
        input_desc, input_data,
        filter_desc, filter_data,
        conv_desc, cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
        workspace_data, workspace_size,
        &c_0,
        output_desc, output_data,
        offset_desc, offset_data,
        relu_desc,
        output_desc, output_data,
    ).is_ok());

    assert!(cuda::cudaMemcpy(
        input.as_mut_ptr() as *mut c_void,
        input_data,
        4 * 25,
        cuda::MemcpyKind::DeviceToHost
    ).is_ok());

    assert!(cuda::cudaMemcpy(
        filter.as_mut_ptr() as *mut c_void,
        filter_data,
        16 * 9,
        cuda::MemcpyKind::DeviceToHost
    ).is_ok());

    assert!(cuda::cudaMemcpy(
        output.as_mut_ptr() as *mut c_void,
        output_data,
        4 * 25,
        cuda::MemcpyKind::DeviceToHost
    ).is_ok());

    //
    let mut plane_1 = vec! [];
    let mut plane_2 = vec! [];
    let mut plane_3 = vec! [];
    let mut plane_4 = vec! [];

    for i in 0..25 {
        plane_1.push(OUTPUT_SCALE * output[4*i+0] as f32 / 127.0);
        plane_2.push(OUTPUT_SCALE * output[4*i+1] as f32 / 127.0);
        plane_3.push(OUTPUT_SCALE * output[4*i+2] as f32 / 127.0);
        plane_4.push(OUTPUT_SCALE * output[4*i+3] as f32 / 127.0);
    }

    println!("");
    println!("input:   {:?}", input);
    println!("filter:  {:?}", filter);
    println!("plane_1: {:?}", plane_1);
    println!("plane_2: {:?}", plane_2);
    println!("plane_3: {:?}", plane_3);
    println!("plane_4: {:?}", plane_4);

    // fact check the output planes with a resolution of:
    //
    // ```
    // (1 / 127) * OUTPUT_SCALE
    // + (1 / 127) * INPUT_SCALE
    // + (1 / 127) * FILTER_SCALE
    // ```
    //
    let eps = (1.0f32 / 127.0) * (OUTPUT_SCALE + INPUT_SCALE + FILTER_SCALE);
    let mut max_error = OrderedFloat(0.0f32);

    let plane_1_1 = 0.01*0.1 + 0.01*0.2 + 0.01*0.3 + 0.01*0.4;
    let plane_1_4 = 0.2 + 4.0 * plane_1_1;  // corner
    let plane_1_6 = 0.2 + 6.0 * plane_1_1;  // side
    let plane_1_9 = 0.2 + 9.0 * plane_1_1;  // middle

    let plane_2_1 = 0.03*0.1 + 0.03*0.2 + 0.03*0.3 + 0.03*0.4;
    let plane_2_4 = 0.1 + 4.0 * plane_2_1;  // corner
    let plane_2_6 = 0.1 + 6.0 * plane_2_1;  // side
    let plane_2_9 = 0.1 + 9.0 * plane_2_1;  // middle

    let plane_3_1 = 0.06*0.1 + 0.06*0.2 + 0.06*0.3 + 0.06*0.4;
    let plane_3_4 = 0.0 + 4.0 * plane_3_1;  // corner
    let plane_3_6 = 0.0 + 6.0 * plane_3_1;  // side
    let plane_3_9 = 0.0 + 9.0 * plane_3_1;  // middle

    let plane_4_1 = 0.09*0.1 + 0.09*0.2 + 0.09*0.3 + 0.09*0.4;
    let plane_4_4 = -0.1 + 4.0 * plane_4_1;  // corner
    let plane_4_6 = -0.1 + 6.0 * plane_4_1;  // side
    let plane_4_9 = -0.1 + 9.0 * plane_4_1;  // middle

    for c in vec! [0, 4, 20, 24].into_iter() {
        assert!(plane_1[c] >= plane_1_4 - eps && plane_1[c] <= plane_1_4 + eps, "plane_1[{}] = {}, should be {}", c, plane_1[c], plane_1_4);
        assert!(plane_2[c] >= plane_2_4 - eps && plane_2[c] <= plane_2_4 + eps, "plane_2[{}] = {}, should be {}", c, plane_2[c], plane_2_4);
        assert!(plane_3[c] >= plane_3_4 - eps && plane_3[c] <= plane_3_4 + eps, "plane_3[{}] = {}, should be {}", c, plane_3[c], plane_3_4);
        assert!(plane_4[c] >= plane_4_4 - eps && plane_4[c] <= plane_4_4 + eps, "plane_4[{}] = {}, should be {}", c, plane_4[c], plane_4_4);

        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_1[c] - plane_1_4).abs()));
        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_2[c] - plane_2_4).abs()));
        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_3[c] - plane_3_4).abs()));
        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_4[c] - plane_4_4).abs()));
    }

    for s in vec! [1, 2, 3, 5, 9, 10, 14, 15, 19, 21, 22, 23].into_iter() {
        assert!(plane_1[s] >= plane_1_6 - eps && plane_1[s] <= plane_1_6 + eps, "plane_1[{}] = {}, should be {}", s, plane_1[s], plane_1_6);
        assert!(plane_2[s] >= plane_2_6 - eps && plane_2[s] <= plane_2_6 + eps, "plane_2[{}] = {}, should be {}", s, plane_2[s], plane_2_6);
        assert!(plane_3[s] >= plane_3_6 - eps && plane_3[s] <= plane_3_6 + eps, "plane_3[{}] = {}, should be {}", s, plane_3[s], plane_3_6);
        assert!(plane_4[s] >= plane_4_6 - eps && plane_4[s] <= plane_4_6 + eps, "plane_4[{}] = {}, should be {}", s, plane_4[s], plane_4_6);

        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_1[s] - plane_1_6).abs()));
        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_2[s] - plane_2_6).abs()));
        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_3[s] - plane_3_6).abs()));
        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_4[s] - plane_4_6).abs()));
    }

    for m in vec! [6, 7, 8, 11, 12, 13, 16, 17, 18].into_iter() {
        assert!(plane_1[m] >= plane_1_9 - eps && plane_1[m] <= plane_1_9 + eps, "plane_1[{}] = {}, should be {}", m, plane_1[m], plane_1_9);
        assert!(plane_2[m] >= plane_2_9 - eps && plane_2[m] <= plane_2_9 + eps, "plane_2[{}] = {}, should be {}", m, plane_2[m], plane_2_9);
        assert!(plane_3[m] >= plane_3_9 - eps && plane_3[m] <= plane_3_9 + eps, "plane_3[{}] = {}, should be {}", m, plane_3[m], plane_3_9);
        assert!(plane_4[m] >= plane_4_9 - eps && plane_4[m] <= plane_4_9 + eps, "plane_4[{}] = {}, should be {}", m, plane_4[m], plane_4_9);

        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_1[m] - plane_1_9).abs()));
        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_2[m] - plane_2_9).abs()));
        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_3[m] - plane_3_9).abs()));
        max_error = ::std::cmp::max(max_error, OrderedFloat((plane_4[m] - plane_4_9).abs()));
    }

    //assert!(max_error <= OrderedFloat(1e-5), "Maximum error is too large! {}", max_error);
}

#[test]
fn _2x0_1() {
    unsafe { __2x0_1() }
}