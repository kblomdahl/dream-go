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

use test::Bencher;
use libc::{c_void};
use std::ptr;

use nn::ffi::*;
use util::f16::*;
use util::q8::*;

#[bench]
fn f32_conv_256(b: &mut Bencher) {
    let mut handle: cudnn::Handle = ptr::null_mut();
    let mut convolution: cudnn::ConvolutionDescriptor = ptr::null_mut();
    let mut filter: cudnn::FilterDescriptor = ptr::null_mut();
    let mut inout_tensor: cudnn::TensorDescriptor = ptr::null_mut();

    let mut input = ptr::null_mut();
    let mut output = ptr::null_mut();
    let mut weights = ptr::null_mut();

    let mut workspace_size: usize = 0;
    let mut workspace = ptr::null_mut();

    unsafe {
        assert!(cudnn::cudnnCreate(&mut handle).is_ok());

        assert!(cudnn::cudnnCreateTensorDescriptor(&mut inout_tensor).is_ok());
        assert!(cudnn::cudnnSetTensor4dDescriptor(
            inout_tensor,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Float,
            1, 256, 19, 19
        ).is_ok());

        assert!(cudnn::cudnnCreateConvolutionDescriptor(&mut convolution).is_ok());
        assert!(cudnn::cudnnSetConvolution2dDescriptor(
            convolution,
            1, 1, 1, 1, 1, 1,
            cudnn::ConvolutionMode::CrossCorrelation,
            cudnn::DataType::Float
        ).is_ok());

        assert!(cudnn::cudnnCreateFilterDescriptor(&mut filter).is_ok());
        assert!(cudnn::cudnnSetFilter4dDescriptor(
            filter,
            cudnn::DataType::Float,
            cudnn::TensorFormat::NCHW,
            256, 256, 3, 3
        ).is_ok());

        assert!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            inout_tensor,
            filter,
            convolution,
            inout_tensor,
            cudnn::ConvolutionFwdAlgo::Winograd,
            &mut workspace_size
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut input, 4 * 92416).is_ok());
        assert!(cuda::cudaMemcpy(
            input,
            vec! [1.0f32; 92416].as_ptr() as *const c_void,
            4 * 92416, cuda::MemcpyKind::HostToDevice
        ).is_ok());
        assert!(cuda::cudaMalloc(&mut weights, 4 * 589824).is_ok());
        assert!(cuda::cudaMemcpy(
            weights,
            vec! [1.0f32; 589824].as_ptr() as *const c_void,
            4 * 589824, cuda::MemcpyKind::HostToDevice
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut output, 4 * 92416).is_ok());
        assert!(cuda::cudaMalloc(&mut workspace, workspace_size).is_ok());

        b.iter(move || {
            let c_0: f32 = 0.0;
            let c_1: f32 = 1.0;

            assert!(cudnn::cudnnConvolutionForward(
                handle,
                &c_1,
                inout_tensor, input,
                filter, weights,
                convolution,
                cudnn::ConvolutionFwdAlgo::Winograd,
                workspace, workspace_size,
                &c_0,
                inout_tensor, output
            ).is_ok());
        });

        // 
        let ones = vec! [0.0f32; 92416];

        assert!(cuda::cudaMemcpy(
            ones.as_ptr() as *mut c_void,
            output,
            4 * 92416, cuda::MemcpyKind::DeviceToHost
        ).is_ok());

        for one in ones.iter() {
            assert!(*one == 1024.0
                || *one == 1536.0
                || *one == 2304.0,
                "{}", *one);
        }
    }
}

#[bench]
fn f16_conv_256(b: &mut Bencher) {
    let mut handle: cudnn::Handle = ptr::null_mut();
    let mut convolution: cudnn::ConvolutionDescriptor = ptr::null_mut();
    let mut filter: cudnn::FilterDescriptor = ptr::null_mut();
    let mut inout_tensor: cudnn::TensorDescriptor = ptr::null_mut();

    let mut input = ptr::null_mut();
    let mut output = ptr::null_mut();
    let mut weights = ptr::null_mut();

    let mut workspace_size: usize = 0;
    let mut workspace = ptr::null_mut();

    unsafe {
        assert!(cudnn::cudnnCreate(&mut handle).is_ok());

        assert!(cudnn::cudnnCreateTensorDescriptor(&mut inout_tensor).is_ok());
        assert!(cudnn::cudnnSetTensor4dDescriptor(
            inout_tensor,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half,
            1, 256, 19, 19
        ).is_ok());

        assert!(cudnn::cudnnCreateConvolutionDescriptor(&mut convolution).is_ok());
        assert!(cudnn::cudnnSetConvolution2dDescriptor(
            convolution,
            1, 1, 1, 1, 1, 1,
            cudnn::ConvolutionMode::CrossCorrelation,
            cudnn::DataType::Half
        ).is_ok());

        #[cfg(feature = "tensor-core")]
        assert!(cudnn::cudnnSetConvolutionMathType(convolution, cudnn::MathType::TensorOpMath).is_ok());

        assert!(cudnn::cudnnCreateFilterDescriptor(&mut filter).is_ok());
        assert!(cudnn::cudnnSetFilter4dDescriptor(
            filter,
            cudnn::DataType::Half,
            cudnn::TensorFormat::NCHW,
            256, 256, 3, 3
        ).is_ok());

        assert!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            inout_tensor,
            filter,
            convolution,
            inout_tensor,
            cudnn::ConvolutionFwdAlgo::WinogradNonFused,
            &mut workspace_size
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut input, 2 * 92416).is_ok());
        assert!(cuda::cudaMemcpy(
            input,
            vec! [f16::from(1.0f32); 92416].as_ptr() as *const c_void,
            2 * 92416, cuda::MemcpyKind::HostToDevice
        ).is_ok());
        assert!(cuda::cudaMalloc(&mut weights, 2 * 589824).is_ok());
        assert!(cuda::cudaMemcpy(
            weights,
            vec! [f16::from(256.0f32.recip()); 589824].as_ptr() as *const c_void,
            2 * 589824, cuda::MemcpyKind::HostToDevice
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut output, 2 * 92416).is_ok());
        assert!(cuda::cudaMalloc(&mut workspace, workspace_size).is_ok());

        b.iter(move || {
            let c_0: f32 = 0.0;
            let c_1: f32 = 1.0;

            assert!(cudnn::cudnnConvolutionForward(
                handle,
                &c_1,
                inout_tensor, input,
                filter, weights,
                convolution,
                cudnn::ConvolutionFwdAlgo::WinogradNonFused,
                workspace, workspace_size,
                &c_0,
                inout_tensor, output
            ).is_ok());
        });

        // 
        let ones = vec! [f16::from(0.0); 92416];

        assert!(cuda::cudaMemcpy(
            ones.as_ptr() as *mut c_void,
            output,
            2 * 92416, cuda::MemcpyKind::DeviceToHost
        ).is_ok());

        for one in ones.iter() {
            let value = f32::from(*one) as i32;

            // f16 is not quite good enough to get
            // exact results so accept any "reasonable"
            // value
            assert!(value >= 3 && value <= 10, "{}", value);
        }
    }
}

#[bench]
fn i8_conv_256(b: &mut Bencher) {
    let mut handle: cudnn::Handle = ptr::null_mut();
    let mut convolution: cudnn::ConvolutionDescriptor = ptr::null_mut();
    let mut filter: cudnn::FilterDescriptor = ptr::null_mut();
    let mut inout_tensor: cudnn::TensorDescriptor = ptr::null_mut();

    let mut input = ptr::null_mut();
    let mut output = ptr::null_mut();
    let mut weights = ptr::null_mut();

    let mut workspace_size: usize = 0;
    let mut workspace = ptr::null_mut();

    unsafe {
        assert!(cudnn::cudnnCreate(&mut handle).is_ok());

        assert!(cudnn::cudnnCreateTensorDescriptor(&mut inout_tensor).is_ok());
        assert!(cudnn::cudnnSetTensor4dDescriptor(
            inout_tensor,
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Int8,
            1, 256, 19, 19
        ).is_ok());

        assert!(cudnn::cudnnCreateConvolutionDescriptor(&mut convolution).is_ok());
        assert!(cudnn::cudnnSetConvolution2dDescriptor(
            convolution,
            1, 1, 1, 1, 1, 1,
            cudnn::ConvolutionMode::CrossCorrelation,
            cudnn::DataType::Int32
        ).is_ok());

        assert!(cudnn::cudnnCreateFilterDescriptor(&mut filter).is_ok());
        assert!(cudnn::cudnnSetFilter4dDescriptor(
            filter,
            cudnn::DataType::Int8,
            cudnn::TensorFormat::NHWC,
            256, 256, 3, 3
        ).is_ok());

        assert!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            inout_tensor,
            filter,
            convolution,
            inout_tensor,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            &mut workspace_size
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut input, 1 * 92416).is_ok());
        assert!(cuda::cudaMemcpy(
            input,
            vec! [q8::from(1.0f32); 92416].as_ptr() as *const c_void,
            1 * 92416, cuda::MemcpyKind::HostToDevice
        ).is_ok());
        assert!(cuda::cudaMalloc(&mut weights, 1 * 589824).is_ok());
        assert!(cuda::cudaMemcpy(
            weights,
            vec! [q8::from(64.0f32.recip()); 589824].as_ptr() as *const c_void,
            1 * 589824, cuda::MemcpyKind::HostToDevice
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut output, 1 * 92416).is_ok());
        assert!(cuda::cudaMalloc(&mut workspace, workspace_size).is_ok());

        b.iter(move || {
            let c_0: f32 = 0.0;
            let c_1: f32 = 1.0;

            assert!(cudnn::cudnnConvolutionForward(
                handle,
                &c_1,
                inout_tensor, input,
                filter, weights,
                convolution,
                cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
                workspace, workspace_size,
                &c_0,
                inout_tensor, output
            ).is_ok());
        });

        // 
        let ones = vec! [q8::from(0.0); 92416];

        assert!(cuda::cudaMemcpy(
            ones.as_ptr() as *mut c_void,
            output,
            1 * 92416, cuda::MemcpyKind::DeviceToHost
        ).is_ok());

        for one in ones.iter() {
            let value = f32::from(*one) as i32;

            assert!(value == 1, "{}", value);
        }
    }
}

#[bench]
fn f32_conv_128(b: &mut Bencher) {
    let mut handle: cudnn::Handle = ptr::null_mut();
    let mut convolution: cudnn::ConvolutionDescriptor = ptr::null_mut();
    let mut filter: cudnn::FilterDescriptor = ptr::null_mut();
    let mut inout_tensor: cudnn::TensorDescriptor = ptr::null_mut();

    let mut input = ptr::null_mut();
    let mut output = ptr::null_mut();
    let mut weights = ptr::null_mut();

    let mut workspace_size: usize = 0;
    let mut workspace = ptr::null_mut();

    unsafe {
        assert!(cudnn::cudnnCreate(&mut handle).is_ok());

        assert!(cudnn::cudnnCreateTensorDescriptor(&mut inout_tensor).is_ok());
        assert!(cudnn::cudnnSetTensor4dDescriptor(
            inout_tensor,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Float,
            1, 128, 19, 19
        ).is_ok());

        assert!(cudnn::cudnnCreateConvolutionDescriptor(&mut convolution).is_ok());
        assert!(cudnn::cudnnSetConvolution2dDescriptor(
            convolution,
            1, 1, 1, 1, 1, 1,
            cudnn::ConvolutionMode::CrossCorrelation,
            cudnn::DataType::Float
        ).is_ok());

        assert!(cudnn::cudnnCreateFilterDescriptor(&mut filter).is_ok());
        assert!(cudnn::cudnnSetFilter4dDescriptor(
            filter,
            cudnn::DataType::Float,
            cudnn::TensorFormat::NCHW,
            128, 128, 3, 3
        ).is_ok());

        assert!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            inout_tensor,
            filter,
            convolution,
            inout_tensor,
            cudnn::ConvolutionFwdAlgo::Winograd,
            &mut workspace_size
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut input, 4 * 46208).is_ok());
        assert!(cuda::cudaMemcpy(
            input,
            vec! [1.0f32; 46208].as_ptr() as *const c_void,
            4 * 46208, cuda::MemcpyKind::HostToDevice
        ).is_ok());
        assert!(cuda::cudaMalloc(&mut weights, 4 * 147456).is_ok());
        assert!(cuda::cudaMemcpy(
            weights,
            vec! [1.0f32; 147456].as_ptr() as *const c_void,
            4 * 147456, cuda::MemcpyKind::HostToDevice
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut output, 4 * 46208).is_ok());
        assert!(cuda::cudaMalloc(&mut workspace, workspace_size).is_ok());

        b.iter(move || {
            let c_0: f32 = 0.0;
            let c_1: f32 = 1.0;

            assert!(cudnn::cudnnConvolutionForward(
                handle,
                &c_1,
                inout_tensor, input,
                filter, weights,
                convolution,
                cudnn::ConvolutionFwdAlgo::Winograd,
                workspace, workspace_size,
                &c_0,
                inout_tensor, output
            ).is_ok());
        });

        // 
        let ones = vec! [0.0f32; 46208];

        assert!(cuda::cudaMemcpy(
            ones.as_ptr() as *mut c_void,
            output,
            4 * 46208, cuda::MemcpyKind::DeviceToHost
        ).is_ok());

        for one in ones.iter() {
            assert!(*one == 512.0
                || *one == 768.0
                || *one == 1152.0,
                "{}", *one);
        }
    }
}

#[bench]
fn f16_conv_128(b: &mut Bencher) {
    let mut handle: cudnn::Handle = ptr::null_mut();
    let mut convolution: cudnn::ConvolutionDescriptor = ptr::null_mut();
    let mut filter: cudnn::FilterDescriptor = ptr::null_mut();
    let mut inout_tensor: cudnn::TensorDescriptor = ptr::null_mut();

    let mut input = ptr::null_mut();
    let mut output = ptr::null_mut();
    let mut weights = ptr::null_mut();

    let mut workspace_size: usize = 0;
    let mut workspace = ptr::null_mut();

    unsafe {
        assert!(cudnn::cudnnCreate(&mut handle).is_ok());

        assert!(cudnn::cudnnCreateTensorDescriptor(&mut inout_tensor).is_ok());
        assert!(cudnn::cudnnSetTensor4dDescriptor(
            inout_tensor,
            cudnn::TensorFormat::NCHW,
            cudnn::DataType::Half,
            1, 128, 19, 19
        ).is_ok());

        assert!(cudnn::cudnnCreateConvolutionDescriptor(&mut convolution).is_ok());
        assert!(cudnn::cudnnSetConvolution2dDescriptor(
            convolution,
            1, 1, 1, 1, 1, 1,
            cudnn::ConvolutionMode::CrossCorrelation,
            cudnn::DataType::Half
        ).is_ok());

        #[cfg(feature = "tensor-core")]
        assert!(cudnn::cudnnSetConvolutionMathType(convolution, cudnn::MathType::TensorOpMath).is_ok());

        assert!(cudnn::cudnnCreateFilterDescriptor(&mut filter).is_ok());
        assert!(cudnn::cudnnSetFilter4dDescriptor(
            filter,
            cudnn::DataType::Half,
            cudnn::TensorFormat::NCHW,
            128, 128, 3, 3
        ).is_ok());

        assert!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            inout_tensor,
            filter,
            convolution,
            inout_tensor,
            cudnn::ConvolutionFwdAlgo::WinogradNonFused,
            &mut workspace_size
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut input, 2 * 46208).is_ok());
        assert!(cuda::cudaMemcpy(
            input,
            vec! [f16::from(1.0f32); 46208].as_ptr() as *const c_void,
            2 * 46208, cuda::MemcpyKind::HostToDevice
        ).is_ok());
        assert!(cuda::cudaMalloc(&mut weights, 2 * 147456).is_ok());
        assert!(cuda::cudaMemcpy(
            weights,
            vec! [f16::from(128.0f32.recip()); 147456].as_ptr() as *const c_void,
            2 * 147456, cuda::MemcpyKind::HostToDevice
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut output, 2 * 46208).is_ok());
        assert!(cuda::cudaMalloc(&mut workspace, workspace_size).is_ok());

        b.iter(move || {
            let c_0: f32 = 0.0;
            let c_1: f32 = 1.0;

            assert!(cudnn::cudnnConvolutionForward(
                handle,
                &c_1,
                inout_tensor, input,
                filter, weights,
                convolution,
                cudnn::ConvolutionFwdAlgo::WinogradNonFused,
                workspace, workspace_size,
                &c_0,
                inout_tensor, output
            ).is_ok());
        });

        // 
        let ones = vec! [f16::from(0.0); 46208];

        assert!(cuda::cudaMemcpy(
            ones.as_ptr() as *mut c_void,
            output,
            2 * 46208, cuda::MemcpyKind::DeviceToHost
        ).is_ok());

        for one in ones.iter() {
            let value = f32::from(*one) as i32;

            // f16 is not quite good enough to get
            // exact results so accept any "reasonable"
            // value
            assert!(value >= 3 && value <= 10, "{}", value);
        }
    }
}

#[bench]
fn i8_conv_128(b: &mut Bencher) {
    let mut handle: cudnn::Handle = ptr::null_mut();
    let mut convolution: cudnn::ConvolutionDescriptor = ptr::null_mut();
    let mut filter: cudnn::FilterDescriptor = ptr::null_mut();
    let mut inout_tensor: cudnn::TensorDescriptor = ptr::null_mut();

    let mut input = ptr::null_mut();
    let mut output = ptr::null_mut();
    let mut weights = ptr::null_mut();

    let mut workspace_size: usize = 0;
    let mut workspace = ptr::null_mut();

    unsafe {
        assert!(cudnn::cudnnCreate(&mut handle).is_ok());

        assert!(cudnn::cudnnCreateTensorDescriptor(&mut inout_tensor).is_ok());
        assert!(cudnn::cudnnSetTensor4dDescriptor(
            inout_tensor,
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Int8,
            1, 128, 19, 19
        ).is_ok());

        assert!(cudnn::cudnnCreateConvolutionDescriptor(&mut convolution).is_ok());
        assert!(cudnn::cudnnSetConvolution2dDescriptor(
            convolution,
            1, 1, 1, 1, 1, 1,
            cudnn::ConvolutionMode::CrossCorrelation,
            cudnn::DataType::Int32
        ).is_ok());

        assert!(cudnn::cudnnCreateFilterDescriptor(&mut filter).is_ok());
        assert!(cudnn::cudnnSetFilter4dDescriptor(
            filter,
            cudnn::DataType::Int8,
            cudnn::TensorFormat::NHWC,
            128, 128, 3, 3
        ).is_ok());

        assert!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            inout_tensor,
            filter,
            convolution,
            inout_tensor,
            cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
            &mut workspace_size
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut input, 1 * 46208).is_ok());
        assert!(cuda::cudaMemcpy(
            input,
            vec! [q8::from(1.0f32); 46208].as_ptr() as *const c_void,
            1 * 46208, cuda::MemcpyKind::HostToDevice
        ).is_ok());
        assert!(cuda::cudaMalloc(&mut weights, 1 * 147456).is_ok());
        assert!(cuda::cudaMemcpy(
            weights,
            vec! [q8::from(64.0f32.recip()); 147456].as_ptr() as *const c_void,
            1 * 147456, cuda::MemcpyKind::HostToDevice
        ).is_ok());

        assert!(cuda::cudaMalloc(&mut output, 1 * 46208).is_ok());
        assert!(cuda::cudaMalloc(&mut workspace, workspace_size).is_ok());

        b.iter(move || {
            let c_0: f32 = 0.0;
            let c_1: f32 = 1.0;

            assert!(cudnn::cudnnConvolutionForward(
                handle,
                &c_1,
                inout_tensor, input,
                filter, weights,
                convolution,
                cudnn::ConvolutionFwdAlgo::ImplicitPrecompGemm,
                workspace, workspace_size,
                &c_0,
                inout_tensor, output
            ).is_ok());
        });

        let ones = vec! [q8::from(0.0); 46208];

        assert!(cuda::cudaMemcpy(
            ones.as_ptr() as *mut c_void,
            output,
            1 * 46208, cuda::MemcpyKind::DeviceToHost
        ).is_ok());

        for one in ones.iter() {
            let value = f32::from(*one) as i32;

            assert!(value == 1, "{}", value);
        }
    }
}