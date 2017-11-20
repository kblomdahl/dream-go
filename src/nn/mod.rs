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

use libc::{c_void};
use std::collections::HashMap;
use std::path::Path;
use std::ptr;

mod cublas;
mod cuda;
mod cudnn;
mod loader;

use self::cublas::*;
use self::cuda::*;
use self::cudnn::*;

const NUM_FEATURES: usize = 256;

pub struct Network {
    handle_blas: cublas::Handle,
    handle_dnn: cudnn::Handle,

    // convolutional descriptors
    conv2d_1: ConvolutionDescriptor,
    conv2d_3: ConvolutionDescriptor,

    // activation functions
    relu: ActivationDescriptor,
    tanh: ActivationDescriptor,

    // filter dimensions
    up_f: FilterDescriptor,
    residual_f: FilterDescriptor,
    value_f: FilterDescriptor,
    policy_f: FilterDescriptor,

    // The array containing the input features.
    input_t: TensorDescriptor,
    residual_t: TensorDescriptor,
    residual_bn_t: TensorDescriptor,
    value_t: TensorDescriptor,
    value_bn_t: TensorDescriptor,
    value_256_t: TensorDescriptor,
    value_1_t: TensorDescriptor,
    policy_t: TensorDescriptor,
    policy_bn_t: TensorDescriptor,
    policy_softmax_t: TensorDescriptor,

    /// The weights of this network
    weights: HashMap<String, *const c_void>,
    zeros: *mut c_void,
    ones: *mut c_void
}

/// A per-thread workspace containing tensor descriptors, operations, and
/// data arrays necessary to perform a forward pass through the neural
/// network.
pub struct Workspace<'a> {
    network: &'a Network,

    input: *mut c_void,
    residual_1: *mut c_void,
    residual_2: *mut c_void,
    residual_3: *mut c_void,
    policy_1: *mut c_void,
    policy_2: *mut c_void,
    value_1: *mut c_void,
    value_2: *mut c_void,

    up_s: usize,
    up_w: *mut c_void,

    residual_s: usize,
    residual_w: *mut c_void,

    value_s: usize,
    value_w: *mut c_void,

    policy_s: usize,
    policy_w: *mut c_void
}

impl Network {
    pub fn new(path: &Path) -> Option<Network> {
        if let Some(weights) = loader::load(path) {
            let mut n = Network {
                handle_blas: ptr::null_mut(),
                handle_dnn: ptr::null_mut(),

                conv2d_1: ptr::null_mut(),
                conv2d_3: ptr::null_mut(),

                relu: ptr::null_mut(),
                tanh: ptr::null_mut(),

                up_f: ptr::null_mut(),
                residual_f: ptr::null_mut(),
                value_f: ptr::null_mut(),
                policy_f: ptr::null_mut(),

                input_t: ptr::null_mut(),
                residual_t: ptr::null_mut(),
                residual_bn_t: ptr::null_mut(),
                value_t: ptr::null_mut(),
                value_bn_t: ptr::null_mut(),
                value_256_t: ptr::null_mut(),
                value_1_t: ptr::null_mut(),
                policy_t: ptr::null_mut(),
                policy_bn_t: ptr::null_mut(),
                policy_softmax_t: ptr::null_mut(),

                weights: weights,
                zeros: ptr::null_mut(),
                ones: ptr::null_mut()
            };

            unsafe {
                assert_eq!(cublasCreate_v2(&mut n.handle_blas), cublas::Status::Success);
                assert_eq!(cudnnCreate(&mut n.handle_dnn), cudnn::Status::Success);

                assert_eq!(cudnnCreateConvolutionDescriptor(&mut n.conv2d_1), cudnn::Status::Success);
                assert_eq!(cudnnSetConvolution2dDescriptor(
                    n.conv2d_1,
                    0, 0,
                    1, 1,
                    1, 1,
                    ConvolutionMode::CrossCorrelation,
                    DataType::Float
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateConvolutionDescriptor(&mut n.conv2d_3), cudnn::Status::Success);
                assert_eq!(cudnnSetConvolution2dDescriptor(
                    n.conv2d_3,
                    1, 1,
                    1, 1,
                    1, 1,
                    ConvolutionMode::CrossCorrelation,
                    DataType::Float
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateActivationDescriptor(&mut n.relu), cudnn::Status::Success);
                assert_eq!(cudnnSetActivationDescriptor(
                    n.relu,
                    ActivationMode::Relu,
                    NanPropagation::PropagateNan,
                    0.0
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateActivationDescriptor(&mut n.tanh), cudnn::Status::Success);
                assert_eq!(cudnnSetActivationDescriptor(
                    n.tanh,
                    ActivationMode::Tanh,
                    NanPropagation::PropagateNan,
                    0.0
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateFilterDescriptor(&mut n.up_f), cudnn::Status::Success);
                assert_eq!(cudnnSetFilter4dDescriptor(
                    n.up_f,
                    DataType::Float,
                    TensorFormat::NCHW,
                    NUM_FEATURES as i32, 34, 3, 3
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateFilterDescriptor(&mut n.residual_f), cudnn::Status::Success);
                assert_eq!(cudnnSetFilter4dDescriptor(
                    n.residual_f,
                    DataType::Float,
                    TensorFormat::NCHW,
                    NUM_FEATURES as i32, NUM_FEATURES as i32, 3, 3
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateFilterDescriptor(&mut n.value_f), cudnn::Status::Success);
                assert_eq!(cudnnSetFilter4dDescriptor(
                    n.value_f,
                    DataType::Float,
                    TensorFormat::NCHW,
                    1, NUM_FEATURES as i32, 1, 1
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateFilterDescriptor(&mut n.policy_f), cudnn::Status::Success);
                assert_eq!(cudnnSetFilter4dDescriptor(
                    n.policy_f,
                    DataType::Float,
                    TensorFormat::NCHW,
                    2, NUM_FEATURES as i32, 1, 1
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateTensorDescriptor(&mut n.input_t), cudnn::Status::Success);
                assert_eq!(cudnnSetTensor4dDescriptor(
                    n.input_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, 34, 19, 19
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateTensorDescriptor(&mut n.residual_t), cudnn::Status::Success);
                assert_eq!(cudnnSetTensor4dDescriptor(
                    n.residual_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, NUM_FEATURES as i32, 19, 19
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateTensorDescriptor(&mut n.residual_bn_t), cudnn::Status::Success);
                assert_eq!(cudnnSetTensor4dDescriptor(
                    n.residual_bn_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, NUM_FEATURES as i32, 1, 1
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateTensorDescriptor(&mut n.value_t), cudnn::Status::Success);
                assert_eq!(cudnnSetTensor4dDescriptor(
                    n.value_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, 1, 19, 19
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateTensorDescriptor(&mut n.value_bn_t), cudnn::Status::Success);
                assert_eq!(cudnnSetTensor4dDescriptor(
                    n.value_bn_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, 1, 1, 1
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateTensorDescriptor(&mut n.value_256_t), cudnn::Status::Success);
                assert_eq!(cudnnSetTensor4dDescriptor(
                    n.value_256_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, NUM_FEATURES as i32, 1, 1
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateTensorDescriptor(&mut n.value_1_t), cudnn::Status::Success);
                assert_eq!(cudnnSetTensor4dDescriptor(
                    n.value_1_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, 1, 1, 1
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateTensorDescriptor(&mut n.policy_t), cudnn::Status::Success);
                assert_eq!(cudnnSetTensor4dDescriptor(
                    n.policy_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, 2, 19, 19
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateTensorDescriptor(&mut n.policy_bn_t), cudnn::Status::Success);
                assert_eq!(cudnnSetTensor4dDescriptor(
                    n.policy_bn_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, 2, 1, 1
                ), cudnn::Status::Success);

                assert_eq!(cudnnCreateTensorDescriptor(&mut n.policy_softmax_t), cudnn::Status::Success);
                assert_eq!(cudnnSetTensor4dDescriptor(
                    n.policy_softmax_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, 362, 1, 1
                ), cudnn::Status::Success);

                assert_eq!(cudaMalloc(&mut n.zeros, 1024), Error::Success);
                assert_eq!(cudaMemcpy(
                    n.zeros,
                    vec! [0.0f32; NUM_FEATURES].as_ptr() as *const c_void,
                    4 * NUM_FEATURES,
                    MemcpyKind::HostToDevice
                ), Error::Success);

                assert_eq!(cudaMalloc(&mut n.ones, 1024), Error::Success);
                assert_eq!(cudaMemcpy(
                    n.ones,
                    vec! [1.0f32; NUM_FEATURES].as_ptr() as *const c_void,
                    4 * NUM_FEATURES,
                    MemcpyKind::HostToDevice
                ), Error::Success);
            }

            Some(n)
        } else {
            None
        }
    }

    /// Returns a structure containing the mutable workspace necessary
    /// to perform a forward pass through the network.
    pub fn get_workspace<'a>(&'a self) -> Workspace<'a> {
        let mut w = Workspace {
            network: self,

            input: ptr::null_mut(),
            residual_1: ptr::null_mut(),
            residual_2: ptr::null_mut(),
            residual_3: ptr::null_mut(),
            policy_1: ptr::null_mut(),
            policy_2: ptr::null_mut(),
            value_1: ptr::null_mut(),
            value_2: ptr::null_mut(),

            up_s: 0,
            up_w: ptr::null_mut(),

            residual_s: 0,
            residual_w: ptr::null_mut(),

            value_s: 0,
            value_w: ptr::null_mut(),

            policy_s: 0,
            policy_w: ptr::null_mut()
        };

        unsafe {
            assert_eq!(cudaMalloc(&mut w.input, 49096), Error::Success);
            assert_eq!(cudaMalloc(&mut w.residual_1, 369664), Error::Success);
            assert_eq!(cudaMalloc(&mut w.residual_2, 369664), Error::Success);
            assert_eq!(cudaMalloc(&mut w.residual_3, 369664), Error::Success);
            assert_eq!(cudaMalloc(&mut w.policy_1, 2888), Error::Success);
            assert_eq!(cudaMalloc(&mut w.policy_2, 2888), Error::Success);
            assert_eq!(cudaMalloc(&mut w.value_1, 1444), Error::Success);
            assert_eq!(cudaMalloc(&mut w.value_2, 1444), Error::Success);

            assert_eq!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
                self.handle_dnn,
                self.input_t,
                self.up_f,
                self.conv2d_3,
                self.residual_t,
                ConvolutionFwdAlgo::Winograd,
                &mut w.up_s
            ), cudnn::Status::Success);

            assert_eq!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
                self.handle_dnn,
                self.residual_t,
                self.residual_f,
                self.conv2d_3,
                self.residual_t,
                ConvolutionFwdAlgo::Winograd,
                &mut w.residual_s
            ), cudnn::Status::Success);

            assert_eq!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
                self.handle_dnn,
                self.residual_t,
                self.value_f,
                self.conv2d_1,
                self.value_t,
                ConvolutionFwdAlgo::ImplicitPrecompGemm,
                &mut w.value_s
            ), cudnn::Status::Success);

            assert_eq!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
                self.handle_dnn,
                self.residual_t,
                self.policy_f,
                self.conv2d_1,
                self.policy_t,
                ConvolutionFwdAlgo::ImplicitPrecompGemm,
                &mut w.policy_s
            ), cudnn::Status::Success);

            assert_eq!(cudaMalloc(&mut w.up_w, w.up_s), Error::Success);
            assert_eq!(cudaMalloc(&mut w.residual_w, w.residual_s), Error::Success);
            assert_eq!(cudaMalloc(&mut w.value_w, w.value_s), Error::Success);
            assert_eq!(cudaMalloc(&mut w.policy_w, w.policy_s), Error::Success);
        }

        w
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        unsafe {
            assert_eq!(cudaFree(self.ones), Error::Success);
            assert_eq!(cudaFree(self.zeros), Error::Success);
            for &v in self.weights.values() {
                assert_eq!(cudaFree(v), Error::Success);
            }

            assert_eq!(cudnnDestroyTensorDescriptor(self.policy_t), cudnn::Status::Success);
            assert_eq!(cudnnDestroyTensorDescriptor(self.policy_bn_t), cudnn::Status::Success);
            assert_eq!(cudnnDestroyTensorDescriptor(self.policy_softmax_t), cudnn::Status::Success);
            assert_eq!(cudnnDestroyTensorDescriptor(self.value_t), cudnn::Status::Success);
            assert_eq!(cudnnDestroyTensorDescriptor(self.value_256_t), cudnn::Status::Success);
            assert_eq!(cudnnDestroyTensorDescriptor(self.value_1_t), cudnn::Status::Success);
            assert_eq!(cudnnDestroyTensorDescriptor(self.value_bn_t), cudnn::Status::Success);
            assert_eq!(cudnnDestroyTensorDescriptor(self.residual_t), cudnn::Status::Success);
            assert_eq!(cudnnDestroyTensorDescriptor(self.residual_bn_t), cudnn::Status::Success);
            assert_eq!(cudnnDestroyTensorDescriptor(self.input_t), cudnn::Status::Success);

            assert_eq!(cudnnDestroyFilterDescriptor(self.up_f), cudnn::Status::Success);
            assert_eq!(cudnnDestroyFilterDescriptor(self.residual_f), cudnn::Status::Success);
            assert_eq!(cudnnDestroyFilterDescriptor(self.policy_f), cudnn::Status::Success);
            assert_eq!(cudnnDestroyFilterDescriptor(self.value_f), cudnn::Status::Success);

            assert_eq!(cudnnDestroyActivationDescriptor(self.tanh), cudnn::Status::Success);
            assert_eq!(cudnnDestroyActivationDescriptor(self.relu), cudnn::Status::Success);

            assert_eq!(cudnnDestroyConvolutionDescriptor(self.conv2d_3), cudnn::Status::Success);
            assert_eq!(cudnnDestroyConvolutionDescriptor(self.conv2d_1), cudnn::Status::Success);

            assert_eq!(cudnnDestroy(self.handle_dnn), cudnn::Status::Success);
            assert_eq!(cublasDestroy_v2(self.handle_blas), cublas::Status::Success);
        }
    }
}

impl<'a> Drop for Workspace<'a> {
    fn drop(&mut self) {
        unsafe {
            assert_eq!(cudaFree(self.input), Error::Success);
            assert_eq!(cudaFree(self.residual_1), Error::Success);
            assert_eq!(cudaFree(self.residual_2), Error::Success);
            assert_eq!(cudaFree(self.residual_3), Error::Success);
            assert_eq!(cudaFree(self.value_1), Error::Success);
            assert_eq!(cudaFree(self.value_2), Error::Success);
            assert_eq!(cudaFree(self.policy_1), Error::Success);
            assert_eq!(cudaFree(self.policy_2), Error::Success);

            assert_eq!(cudaFree(self.up_w), Error::Success);
            assert_eq!(cudaFree(self.residual_w), Error::Success);
            assert_eq!(cudaFree(self.policy_w), Error::Success);
            assert_eq!(cudaFree(self.value_w), Error::Success);
        }
    }
}

/// Returns the value and policy tensors obtained from a forward pass
/// through the neural network.
///
/// # Arguments
///
/// * `ws` - the workspace for the current thread
/// * `features` - the input features
///
pub fn forward(w: &mut Workspace, features: &[f32]) -> (f32, Box<[f32]>) {
    let epsilon: f64 = 1e-5;
    let c_0 = 0.0f32;
    let c_1 = 1.0f32;

    let mut softmax = vec! [0.0f32; 362];
    let mut value = vec! [0.0f32; 1];

    unsafe {
        assert_eq!(cudaMemcpy(
            w.input,
            features.as_ptr() as *const c_void,
            12274,
            MemcpyKind::HostToDevice
        ), Error::Success);

        // up-sample the input features to the 256-wide internal representation
        assert_eq!(cudnnConvolutionForward(
            w.network.handle_dnn,
            &c_1,  // alpha
            w.network.input_t, w.input,  // input
            w.network.up_f, w.network.weights["01_upsample/weights:0"],  // weights
            w.network.conv2d_3,  // convolution
            ConvolutionFwdAlgo::Winograd,  // algo
            w.up_w, w.up_s,  // workspace
            &c_0,  // beta
            w.network.residual_t, w.residual_1,  // output
        ), cudnn::Status::Success);

        assert_eq!(cudnnBatchNormalizationForwardInference(
            w.network.handle_dnn,
            BatchNormMode::Spatial,
            &c_1,  // alpha
            &c_0,  // beta
            w.network.residual_t, w.residual_1,  // input
            w.network.residual_t, w.residual_2,  // output
            w.network.residual_bn_t,
            w.network.ones, w.network.zeros,  // scale, bias
            w.network.weights["01_upsample/mean:0"],
            w.network.weights["01_upsample/variance:0"],
            epsilon
        ), cudnn::Status::Success);

        assert_eq!(cudnnActivationForward(
            w.network.handle_dnn,
            w.network.relu,
            &c_1,  // alpha
            w.network.residual_t, w.residual_2,  // input
            &c_0,  // beta
            w.network.residual_t, w.residual_1,  // output
        ), cudnn::Status::Success);

        // apply all of the residual blocks
        for i in 2..21 {
            assert_eq!(cudnnConvolutionForward(
                w.network.handle_dnn,
                &c_1,  // alpha
                w.network.residual_t, w.residual_1,  // input
                w.network.residual_f, w.network.weights[&format!("{:02}_residual/weights_1:0", i)],  // weights
                w.network.conv2d_3,  // convolution
                ConvolutionFwdAlgo::Winograd,  // algo
                w.residual_w, w.residual_s,  // workspace
                &c_0,  // beta
                w.network.residual_t, w.residual_2,  // output
            ), cudnn::Status::Success);

            assert_eq!(cudnnBatchNormalizationForwardInference(
                w.network.handle_dnn,
                BatchNormMode::Spatial,
                &c_1,  // alpha
                &c_0,  // beta
                w.network.residual_t, w.residual_2,  // input
                w.network.residual_t, w.residual_3,  // output
                w.network.residual_bn_t,
                w.network.ones, w.network.zeros,  // scale, bias
                w.network.weights[&format!("{:02}_residual/mean_1:0", i)],
                w.network.weights[&format!("{:02}_residual/variance_1:0", i)],
                epsilon
            ), cudnn::Status::Success);

            assert_eq!(cudnnActivationForward(
                w.network.handle_dnn,
                w.network.relu,
                &c_1,  // alpha
                w.network.residual_t, w.residual_3,  // input
                &c_0,  // beta
                w.network.residual_t, w.residual_3,  // output
            ), cudnn::Status::Success);

            assert_eq!(cudnnConvolutionForward(
                w.network.handle_dnn,
                &c_1,  // alpha
                w.network.residual_t, w.residual_3,  // input
                w.network.residual_f, w.network.weights[&format!("{:02}_residual/weights_2:0", i)],  // weights
                w.network.conv2d_3,  // convolution
                ConvolutionFwdAlgo::Winograd,  // algo
                w.residual_w, w.residual_s,  // workspace
                &c_0,  // beta
                w.network.residual_t, w.residual_2,  // output
            ), cudnn::Status::Success);

            assert_eq!(cudnnBatchNormalizationForwardInference(
                w.network.handle_dnn,
                BatchNormMode::Spatial,
                &c_1,  // alpha
                &c_1,  // beta
                w.network.residual_t, w.residual_2,  // input
                w.network.residual_t, w.residual_1,  // output
                w.network.residual_bn_t,
                w.network.ones, w.network.zeros,  // scale, bias
                w.network.weights[&format!("{:02}_residual/mean_2:0", i)],
                w.network.weights[&format!("{:02}_residual/variance_2:0", i)],
                epsilon
            ), cudnn::Status::Success);

            assert_eq!(cudnnActivationForward(
                w.network.handle_dnn,
                w.network.relu,
                &c_1,  // alpha
                w.network.residual_t, w.residual_1,  // input
                &c_0,  // beta
                w.network.residual_t, w.residual_1,  // output
            ), cudnn::Status::Success);
        }

        // policy head (21p_policy)
        assert_eq!(cudnnConvolutionForward(
            w.network.handle_dnn,
            &c_1,  // alpha
            w.network.residual_t, w.residual_1,  // input
            w.network.policy_f, w.network.weights["21p_policy/downsample:0"],  // weights
            w.network.conv2d_1,  // convolution
            ConvolutionFwdAlgo::ImplicitPrecompGemm,  // algo
            w.policy_w, w.policy_s,  // workspace
            &c_0,  // beta
            w.network.policy_t, w.policy_1,  // output
        ), cudnn::Status::Success);

        assert_eq!(cudnnBatchNormalizationForwardInference(
            w.network.handle_dnn,
            BatchNormMode::Spatial,
            &c_1,  // alpha
            &c_0,  // beta
            w.network.policy_t, w.policy_1,  // input
            w.network.policy_t, w.policy_2,  // output
            w.network.policy_bn_t,
            w.network.ones, w.network.zeros,  // scale, bias
            w.network.weights["21p_policy/mean:0"],
            w.network.weights["21p_policy/variance:0"],
            epsilon
        ), cudnn::Status::Success);

        assert_eq!(cudnnActivationForward(
            w.network.handle_dnn,
            w.network.relu,
            &c_1,  // alpha
            w.network.policy_t, w.policy_2,  // input
            &c_0,  // beta
            w.network.policy_t, w.policy_2,  // output
        ), cudnn::Status::Success);

        assert_eq!(cublasSgemm_v2(
            w.network.handle_blas,
            Operation::T,
            Operation::T,
            362, 1, 722, // output_dims, batch_size, input_dims
            &c_1,  // alpha
            w.network.weights["21p_policy/weights:0"], 722,  // input_2
            w.policy_2, 1,  // input_1
            &c_0,  // beta
            w.policy_1, 362  // output
        ), cublas::Status::Success);

        assert_eq!(cudnnAddTensor(
            w.network.handle_dnn,
            &c_1,  // alpha
            w.network.policy_softmax_t, w.network.weights["21p_policy/bias:0"],  // bias
            &c_1,  // beta
            w.network.policy_softmax_t, w.policy_1  // input and output
        ), cudnn::Status::Success);

        assert_eq!(cudnnSoftmaxForward(
            w.network.handle_dnn,
            SoftmaxAlgorithm::Accurate,
            SoftmaxMode::Instance,
            &c_1,  // alpha
            w.network.policy_softmax_t, w.policy_1,  // input
            &c_0,  // beta
            w.network.policy_softmax_t, w.policy_2  // output
        ), cudnn::Status::Success);

        assert_eq!(cudaMemcpy(
            softmax.as_mut_ptr() as *mut c_void,
            w.policy_2,
            1448,
            MemcpyKind::DeviceToHost
        ), Error::Success);

        // value head (21v_value)
        assert_eq!(cudnnConvolutionForward(
            w.network.handle_dnn,
            &c_1,  // alpha
            w.network.residual_t, w.residual_1,  // input
            w.network.value_f, w.network.weights["21v_value/downsample:0"],  // weights
            w.network.conv2d_1,  // convolution
            ConvolutionFwdAlgo::ImplicitPrecompGemm,  // algo
            w.value_w, w.value_s,  // workspace
            &c_0,  // beta
            w.network.value_t, w.value_1  // output
        ), cudnn::Status::Success);

        assert_eq!(cudnnBatchNormalizationForwardInference(
            w.network.handle_dnn,
            BatchNormMode::Spatial,
            &c_1,  // alpha
            &c_0,  // beta
            w.network.value_t, w.value_1,  // input
            w.network.value_t, w.value_2,  // output
            w.network.value_bn_t,
            w.network.ones, w.network.zeros,  // scale, bias
            w.network.weights["21v_value/mean:0"],
            w.network.weights["21v_value/variance:0"],
            epsilon
        ), cudnn::Status::Success);

        assert_eq!(cudnnActivationForward(
            w.network.handle_dnn,
            w.network.relu,
            &c_1,  // alpha
            w.network.value_t, w.value_2,  // input
            &c_0,  // beta
            w.network.value_t, w.value_2,  // output
        ), cudnn::Status::Success);

        assert_eq!(cublasSgemm_v2(
            w.network.handle_blas,
            Operation::T,
            Operation::T,
            256, 1, 361,  // output_dims, batch_size, input_dims
            &c_1,  // alpha
            w.network.weights["21v_value/weights_1:0"], 361,  // input_2
            w.value_2, 1,  // input_1
            &c_0,  // beta
            w.value_1, 256  // output
        ), cublas::Status::Success);

        assert_eq!(cudnnAddTensor(
            w.network.handle_dnn,
            &c_1,  // alpha
            w.network.value_256_t, w.network.weights["21v_value/bias_1:0"],  // bias
            &c_1,  // beta
            w.network.value_256_t, w.value_1  // input and output
        ), cudnn::Status::Success);

        assert_eq!(cudnnActivationForward(
            w.network.handle_dnn,
            w.network.relu,
            &c_1,  // alpha
            w.network.value_256_t, w.value_1,  // input
            &c_0,  // beta
            w.network.value_256_t, w.value_1,  // output
        ), cudnn::Status::Success);

        assert_eq!(cublasSgemm_v2(
            w.network.handle_blas,
            Operation::T,
            Operation::T,
            1, 1, 256,  // output_dims, batch_size, input_dims
            &c_1,  // alpha
            w.network.weights["21v_value/weights_2:0"], 256,  // input_2
            w.value_1, 1,  // input_1
            &c_0,  // beta
            w.value_2, 1  // output
        ), cublas::Status::Success);

        assert_eq!(cudnnAddTensor(
            w.network.handle_dnn,
            &c_1,  // alpha
            w.network.value_1_t, w.network.weights["21v_value/bias_2:0"],  // bias
            &c_1,  // beta
            w.network.value_1_t, w.value_2  // input and output
        ), cudnn::Status::Success);

        assert_eq!(cudnnActivationForward(
            w.network.handle_dnn,
            w.network.tanh,
            &c_1,  // alpha
            w.network.value_1_t, w.value_2,  // input
            &c_0,  // beta
            w.network.value_1_t, w.value_2,  // output
        ), cudnn::Status::Success);

        assert_eq!(cudaMemcpy(
            value.as_mut_ptr() as *mut c_void,
            w.value_2,
            4,
            MemcpyKind::DeviceToHost
        ), Error::Success);
    }

    (value[0], softmax.into_boxed_slice())
}
