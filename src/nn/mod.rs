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

mod ffi;
mod loader;

use libc::{c_void};
use std::collections::HashMap;
use std::path::Path;
use std::ptr;

use self::ffi::cublas::*;
use self::ffi::cuda::*;
use self::ffi::cudnn::*;

/// The width of each filter in the neural network. Larger is "better" but takes
/// longer to train and gives worse runtime performance during inference.
const NUM_FEATURES: usize = 256;

pub struct Network {
    handle_blas: self::ffi::cublas::Handle,
    handle_dnn: self::ffi::cudnn::Handle,

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

    // bias tensors
    policy_bias_t: TensorDescriptor,
    value_256_bias_t: TensorDescriptor,
    value_1_bias_t: TensorDescriptor,

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
    handle_blas: self::ffi::cublas::Handle,
    handle_dnn: self::ffi::cudnn::Handle,
    batch_size: usize,

    tower_s: Stream,
    policy_s: Stream,
    value_s: Stream,
    tower_e: Event,

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

    input: *mut c_void,
    residual_1: *mut c_void,
    residual_2: *mut c_void,
    residual_3: *mut c_void,
    policy_1: *mut c_void,
    policy_2: *mut c_void,
    value_1: *mut c_void,
    value_2: *mut c_void,

    scratch_size: usize,
    scratch_1: *mut c_void,
    scratch_2: *mut c_void
}

macro_rules! check {
    ($status:expr) => ({
        let err = $status;

        assert!(err.is_ok(), "cuda call failed -- {:?}", err);
    });

    ($status:expr, $name:expr, $result:expr, $batch_size:expr, $n:expr) => ({
        check!($status);

        #[cfg(feature = "trace-cuda")]
        {
            // copy the memory from the device back to the host so
            // that we can look at it
            let mut vec = vec! [[0.0f32; $n]; $batch_size];

            check!(cudaDeviceSynchronize());
            check!(cudaMemcpy(
                vec.as_mut_ptr() as *mut c_void,
                $result,
                4 * $batch_size * $n,
                MemcpyKind::DeviceToHost
            ));

            // pretty-print the array and then output the debugging
            // information
            let mut s = String::new();
            let mut sum = 0.0f32;

            for i in 0..$batch_size {
                if i > 0 { s += ", "; }
                s += "[";

                if $n < 8 {
                    for j in 0..$n {
                        if j > 0 { s += ", "; }

                        sum += vec[i][j];
                        s += &format!("{:.4}", vec[i][j]);
                    }
                } else {
                    for j in 0..8 {
                        s += &format!("{:.4}, ", vec[i][j]);
                    }

                    s += "...";
                }

                s += "]";
            }

            println!("sum(`{}`) == {}", $name, sum);
            println!("eval(`{}`) == [{}]", $name, s);
        }
    });

    ($status:expr, $name:expr, $result:expr, $batch_size:expr, $m:expr, $n:expr) => ({
        check!($status, $name, $result, $batch_size, $m * $n);
    })
}

impl Network {
    pub fn new(path: &Path) -> Option<Network> {
        unsafe {
            check!(cuInit(0));
        }

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

                policy_bias_t: ptr::null_mut(),
                value_256_bias_t: ptr::null_mut(),
                value_1_bias_t: ptr::null_mut(),

                weights: weights,
                zeros: ptr::null_mut(),
                ones: ptr::null_mut()
            };

            unsafe {
                check!(cublasCreate_v2(&mut n.handle_blas));
                check!(cudnnCreate(&mut n.handle_dnn));

                check!(cudnnCreateConvolutionDescriptor(&mut n.conv2d_1));
                check!(cudnnSetConvolution2dDescriptor(
                    n.conv2d_1,
                    0, 0,
                    1, 1,
                    1, 1,
                    ConvolutionMode::CrossCorrelation,
                    DataType::Float
                ));

                check!(cudnnCreateConvolutionDescriptor(&mut n.conv2d_3));
                check!(cudnnSetConvolution2dDescriptor(
                    n.conv2d_3,
                    1, 1,
                    1, 1,
                    1, 1,
                    ConvolutionMode::CrossCorrelation,
                    DataType::Float
                ));

                check!(cudnnCreateActivationDescriptor(&mut n.relu));
                check!(cudnnSetActivationDescriptor(
                    n.relu,
                    ActivationMode::Relu,
                    NanPropagation::PropagateNan,
                    0.0
                ));

                check!(cudnnCreateActivationDescriptor(&mut n.tanh));
                check!(cudnnSetActivationDescriptor(
                    n.tanh,
                    ActivationMode::Tanh,
                    NanPropagation::PropagateNan,
                    0.0
                ));

                check!(cudnnCreateFilterDescriptor(&mut n.up_f));
                check!(cudnnSetFilter4dDescriptor(
                    n.up_f,
                    DataType::Float,
                    TensorFormat::NCHW,
                    NUM_FEATURES as i32, 34, 3, 3
                ));

                check!(cudnnCreateFilterDescriptor(&mut n.residual_f));
                check!(cudnnSetFilter4dDescriptor(
                    n.residual_f,
                    DataType::Float,
                    TensorFormat::NCHW,
                    NUM_FEATURES as i32, NUM_FEATURES as i32, 3, 3
                ));

                check!(cudnnCreateFilterDescriptor(&mut n.value_f));
                check!(cudnnSetFilter4dDescriptor(
                    n.value_f,
                    DataType::Float,
                    TensorFormat::NCHW,
                    1, NUM_FEATURES as i32, 1, 1
                ));

                check!(cudnnCreateFilterDescriptor(&mut n.policy_f));
                check!(cudnnSetFilter4dDescriptor(
                    n.policy_f,
                    DataType::Float,
                    TensorFormat::NCHW,
                    2, NUM_FEATURES as i32, 1, 1
                ));

                check!(cudnnCreateTensorDescriptor(&mut n.policy_bias_t));
                check!(cudnnSetTensor4dDescriptor(
                    n.policy_bias_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, 362, 1, 1
                ));

                check!(cudnnCreateTensorDescriptor(&mut n.value_256_bias_t));
                check!(cudnnSetTensor4dDescriptor(
                    n.value_256_bias_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, 256, 1, 1
                ));

                check!(cudnnCreateTensorDescriptor(&mut n.value_1_bias_t));
                check!(cudnnSetTensor4dDescriptor(
                    n.value_1_bias_t,
                    TensorFormat::NCHW,
                    DataType::Float,
                    1, 1, 1, 1
                ));

                check!(cudaMalloc(&mut n.zeros, 1024));
                check!(cudaMemcpy(
                    n.zeros,
                    vec! [0.0f32; NUM_FEATURES].as_ptr() as *const c_void,
                    4 * NUM_FEATURES,
                    MemcpyKind::HostToDevice
                ));

                check!(cudaMalloc(&mut n.ones, 1024));
                check!(cudaMemcpy(
                    n.ones,
                    vec! [1.0f32; NUM_FEATURES].as_ptr() as *const c_void,
                    4 * NUM_FEATURES,
                    MemcpyKind::HostToDevice
                ));
            }

            Some(n)
        } else {
            None
        }
    }

    /// Returns a structure containing the mutable workspace necessary
    /// to perform a forward pass through the network.
    pub fn get_workspace<'a>(&'a self, batch_size: usize) -> Workspace<'a> {
        assert!(batch_size >= 1);

        let mut w = Workspace {
            network: self,
            handle_blas: ptr::null_mut(),
            handle_dnn: ptr::null_mut(),
            batch_size: batch_size,

            tower_s: ptr::null_mut(),
            policy_s: ptr::null_mut(),
            value_s: ptr::null_mut(),
            tower_e: ptr::null_mut(),

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

            input: ptr::null_mut(),
            residual_1: ptr::null_mut(),
            residual_2: ptr::null_mut(),
            residual_3: ptr::null_mut(),
            policy_1: ptr::null_mut(),
            policy_2: ptr::null_mut(),
            value_1: ptr::null_mut(),
            value_2: ptr::null_mut(),

            scratch_size: 0,
            scratch_1: ptr::null_mut(),
            scratch_2: ptr::null_mut(),
        };

        unsafe {
            check!(cublasCreate_v2(&mut w.handle_blas));
            check!(cudnnCreate(&mut w.handle_dnn));

            check!(cudaStreamCreate(&mut w.tower_s));
            check!(cudaStreamCreate(&mut w.policy_s));
            check!(cudaStreamCreate(&mut w.value_s));
            check!(cudaEventCreate(&mut w.tower_e));

            check!(cudnnCreateTensorDescriptor(&mut w.input_t));
            check!(cudnnSetTensor4dDescriptor(
                w.input_t,
                TensorFormat::NCHW,
                DataType::Float,
                batch_size as i32, 34, 19, 19
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.residual_t));
            check!(cudnnSetTensor4dDescriptor(
                w.residual_t,
                TensorFormat::NCHW,
                DataType::Float,
                batch_size as i32, NUM_FEATURES as i32, 19, 19
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.residual_bn_t));
            check!(cudnnSetTensor4dDescriptor(
                w.residual_bn_t,
                TensorFormat::NCHW,
                DataType::Float,
                1, NUM_FEATURES as i32, 1, 1
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.value_t));
            check!(cudnnSetTensor4dDescriptor(
                w.value_t,
                TensorFormat::NCHW,
                DataType::Float,
                batch_size as i32, 1, 19, 19
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.value_bn_t));
            check!(cudnnSetTensor4dDescriptor(
                w.value_bn_t,
                TensorFormat::NCHW,
                DataType::Float,
                1, 1, 1, 1
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.value_256_t));
            check!(cudnnSetTensor4dDescriptor(
                w.value_256_t,
                TensorFormat::NCHW,
                DataType::Float,
                batch_size as i32, NUM_FEATURES as i32, 1, 1
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.value_1_t));
            check!(cudnnSetTensor4dDescriptor(
                w.value_1_t,
                TensorFormat::NCHW,
                DataType::Float,
                batch_size as i32, 1, 1, 1
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.policy_t));
            check!(cudnnSetTensor4dDescriptor(
                w.policy_t,
                TensorFormat::NCHW,
                DataType::Float,
                batch_size as i32, 2, 19, 19
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.policy_bn_t));
            check!(cudnnSetTensor4dDescriptor(
                w.policy_bn_t,
                TensorFormat::NCHW,
                DataType::Float,
                1, 2, 1, 1
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.policy_softmax_t));
            check!(cudnnSetTensor4dDescriptor(
                w.policy_softmax_t,
                TensorFormat::NCHW,
                DataType::Float,
                batch_size as i32, 362, 1, 1
            ));

            check!(cudaMalloc(&mut w.input, batch_size * 49096));
            check!(cudaMalloc(&mut w.residual_1, batch_size * 369664));
            check!(cudaMalloc(&mut w.residual_2, batch_size * 369664));
            check!(cudaMalloc(&mut w.residual_3, batch_size * 369664));
            check!(cudaMalloc(&mut w.policy_1, batch_size * 2888));
            check!(cudaMalloc(&mut w.policy_2, batch_size * 2888));
            check!(cudaMalloc(&mut w.value_1, batch_size * 1444));
            check!(cudaMalloc(&mut w.value_2, batch_size * 1444));

            // allocate two scratch workspaces that are at least as large as the
            // largest workspace requested by cuDNN.
            let mut up_s: usize = 0;
            let mut residual_s: usize = 0;
            let mut value_s: usize = 0;
            let mut policy_s: usize = 0;

            check!(cudnnGetConvolutionForwardWorkspaceSize(
                self.handle_dnn,
                w.input_t,
                self.up_f,
                self.conv2d_3,
                w.residual_t,
                ConvolutionFwdAlgo::Winograd,
                &mut up_s
            ));

            check!(cudnnGetConvolutionForwardWorkspaceSize(
                self.handle_dnn,
                w.residual_t,
                self.residual_f,
                self.conv2d_3,
                w.residual_t,
                ConvolutionFwdAlgo::Winograd,
                &mut residual_s
            ));

            check!(cudnnGetConvolutionForwardWorkspaceSize(
                self.handle_dnn,
                w.residual_t,
                self.value_f,
                self.conv2d_1,
                w.value_t,
                ConvolutionFwdAlgo::ImplicitPrecompGemm,
                &mut value_s
            ));

            check!(cudnnGetConvolutionForwardWorkspaceSize(
                self.handle_dnn,
                w.residual_t,
                self.policy_f,
                self.conv2d_1,
                w.policy_t,
                ConvolutionFwdAlgo::ImplicitPrecompGemm,
                &mut policy_s
            ));

            w.scratch_size = vec! [up_s, residual_s, value_s, policy_s].into_iter().max().unwrap();

            check!(cudaMalloc(&mut w.scratch_1, w.scratch_size));
            check!(cudaMalloc(&mut w.scratch_2, w.scratch_size));
        }

        w
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        unsafe {
            check!(cudaFree(self.ones));
            check!(cudaFree(self.zeros));
            for &v in self.weights.values() {
                check!(cudaFree(v));
            }

            check!(cudnnDestroyTensorDescriptor(self.value_1_bias_t));
            check!(cudnnDestroyFilterDescriptor(self.value_256_bias_t));
            check!(cudnnDestroyFilterDescriptor(self.policy_bias_t));

            check!(cudnnDestroyFilterDescriptor(self.up_f));
            check!(cudnnDestroyFilterDescriptor(self.residual_f));
            check!(cudnnDestroyFilterDescriptor(self.policy_f));
            check!(cudnnDestroyFilterDescriptor(self.value_f));

            check!(cudnnDestroyActivationDescriptor(self.tanh));
            check!(cudnnDestroyActivationDescriptor(self.relu));

            check!(cudnnDestroyConvolutionDescriptor(self.conv2d_3));
            check!(cudnnDestroyConvolutionDescriptor(self.conv2d_1));

            check!(cudnnDestroy(self.handle_dnn));
            check!(cublasDestroy_v2(self.handle_blas));
        }
    }
}

impl<'a> Drop for Workspace<'a> {
    fn drop(&mut self) {
        unsafe {
            check!(cudaFree(self.scratch_1));
            check!(cudaFree(self.scratch_2));

            check!(cudaFree(self.input));
            check!(cudaFree(self.residual_1));
            check!(cudaFree(self.residual_2));
            check!(cudaFree(self.residual_3));
            check!(cudaFree(self.value_1));
            check!(cudaFree(self.value_2));
            check!(cudaFree(self.policy_1));
            check!(cudaFree(self.policy_2));

            check!(cudnnDestroyTensorDescriptor(self.policy_t));
            check!(cudnnDestroyTensorDescriptor(self.policy_bn_t));
            check!(cudnnDestroyTensorDescriptor(self.policy_softmax_t));
            check!(cudnnDestroyTensorDescriptor(self.value_t));
            check!(cudnnDestroyTensorDescriptor(self.value_256_t));
            check!(cudnnDestroyTensorDescriptor(self.value_1_t));
            check!(cudnnDestroyTensorDescriptor(self.value_bn_t));
            check!(cudnnDestroyTensorDescriptor(self.residual_t));
            check!(cudnnDestroyTensorDescriptor(self.residual_bn_t));
            check!(cudnnDestroyTensorDescriptor(self.input_t));

            check!(cudaEventDestroy(self.tower_e));
            check!(cudaStreamDestroy(self.tower_s));
            check!(cudaStreamDestroy(self.policy_s));
            check!(cudaStreamDestroy(self.value_s));

            check!(cudnnDestroy(self.handle_dnn));
            check!(cublasDestroy_v2(self.handle_blas));
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
pub fn forward(w: &mut Workspace, features: &Vec<Box<[f32]>>) -> (Vec<f32>, Vec<Box<[f32]>>) {
    assert_eq!(w.batch_size, features.len());

    let epsilon: f64 = 0.001;  // tensorflow default
    let c_0 = 0.0f32;
    let c_1 = 1.0f32;

    let mut softmax = vec! [vec! [0.0f32; 362]; w.batch_size];
    let mut value = vec! [0.0f32; w.batch_size];

    unsafe {
        check!(cudnnSetStream(w.handle_dnn, w.tower_s));

        for (i, ref feature) in features.iter().enumerate() {
            assert_eq!(feature.len(), 12274);
            assert_eq!(1, ::std::mem::size_of::<c_void>());

            check!(cudaMemcpyAsync(
                w.input.offset((i * 49096) as isize),
                feature.as_ptr() as *const c_void,
                49096,
                MemcpyKind::HostToDevice,
                w.tower_s
            ));
        }

        // up-sample the input features to the 256-wide internal representation
        check!(cudnnConvolutionForward(
            w.handle_dnn,
            &c_1,  // alpha
            w.input_t, w.input,  // input
            w.network.up_f, w.network.weights["01_upsample/weights:0"],  // weights
            w.network.conv2d_3,  // convolution
            ConvolutionFwdAlgo::Winograd,  // algo
            w.scratch_1, w.scratch_size,  // workspace
            &c_0,  // beta
            w.residual_t, w.residual_1,  // output
        ), "01_upsample/up", w.residual_1, w.batch_size, NUM_FEATURES, 361);

        check!(cudnnBatchNormalizationForwardInference(
            w.handle_dnn,
            BatchNormMode::Spatial,
            &c_1,  // alpha
            &c_0,  // beta
            w.residual_t, w.residual_1,  // input
            w.residual_t, w.residual_2,  // output
            w.residual_bn_t,
            w.network.ones, w.network.zeros,  // scale, bias
            w.network.weights["01_upsample/mean:0"],
            w.network.weights["01_upsample/variance:0"],
            epsilon
        ), "01_upsample/up_bn", w.residual_2, w.batch_size, NUM_FEATURES, 361);

        check!(cudnnActivationForward(
            w.handle_dnn,
            w.network.relu,
            &c_1,  // alpha
            w.residual_t, w.residual_2,  // input
            &c_0,  // beta
            w.residual_t, w.residual_1,  // output
        ), "01_upsample/up_relu", w.residual_1, w.batch_size, NUM_FEATURES, 361);

        // apply all of the residual blocks
        for i in 2..21 {
            check!(cudnnConvolutionForward(
                w.handle_dnn,
                &c_1,  // alpha
                w.residual_t, w.residual_1,  // input
                w.network.residual_f, w.network.weights[&format!("{:02}_residual/weights_1:0", i)],  // weights
                w.network.conv2d_3,  // convolution
                ConvolutionFwdAlgo::Winograd,  // algo
                w.scratch_1, w.scratch_size,  // workspace
                &c_0,  // beta
                w.residual_t, w.residual_2,  // output
            ), &format!("{:02}_residual/conv_1", i), w.residual_2, w.batch_size, NUM_FEATURES, 361);

            check!(cudnnBatchNormalizationForwardInference(
                w.handle_dnn,
                BatchNormMode::Spatial,
                &c_1,  // alpha
                &c_0,  // beta
                w.residual_t, w.residual_2,  // input
                w.residual_t, w.residual_3,  // output
                w.residual_bn_t,
                w.network.ones, w.network.zeros,  // scale, bias
                w.network.weights[&format!("{:02}_residual/mean_1:0", i)],
                w.network.weights[&format!("{:02}_residual/variance_1:0", i)],
                epsilon
            ), &format!("{:02}_residual/conv_bn_1", i), w.residual_3, w.batch_size, NUM_FEATURES, 361);

            check!(cudnnActivationForward(
                w.handle_dnn,
                w.network.relu,
                &c_1,  // alpha
                w.residual_t, w.residual_3,  // input
                &c_0,  // beta
                w.residual_t, w.residual_3,  // output
            ), &format!("{:02}_residual/conv_relu_1", i), w.residual_3, w.batch_size, NUM_FEATURES, 361);

            check!(cudnnConvolutionForward(
                w.handle_dnn,
                &c_1,  // alpha
                w.residual_t, w.residual_3,  // input
                w.network.residual_f, w.network.weights[&format!("{:02}_residual/weights_2:0", i)],  // weights
                w.network.conv2d_3,  // convolution
                ConvolutionFwdAlgo::Winograd,  // algo
                w.scratch_1, w.scratch_size,  // workspace
                &c_0,  // beta
                w.residual_t, w.residual_2,  // output
            ), &format!("{:02}_residual/conv_2", i), w.residual_2, w.batch_size, NUM_FEATURES, 361);

            check!(cudnnBatchNormalizationForwardInference(
                w.handle_dnn,
                BatchNormMode::Spatial,
                &c_1,  // alpha
                &c_1,  // beta
                w.residual_t, w.residual_2,  // input
                w.residual_t, w.residual_1,  // output
                w.residual_bn_t,
                w.network.ones, w.network.zeros,  // scale, bias
                w.network.weights[&format!("{:02}_residual/mean_2:0", i)],
                w.network.weights[&format!("{:02}_residual/variance_2:0", i)],
                epsilon
            ), &format!("{:02}_residual/conv_bn_2", i), w.residual_1, w.batch_size, NUM_FEATURES, 361);

            check!(cudnnActivationForward(
                w.handle_dnn,
                w.network.relu,
                &c_1,  // alpha
                w.residual_t, w.residual_1,  // input
                &c_0,  // beta
                w.residual_t, w.residual_1,  // output
            ), &format!("{:02}_residual/conv_relu_2", i), w.residual_1, w.batch_size, NUM_FEATURES, 361);
        }

        check!(cudaEventRecord(w.tower_e, w.tower_s));
        check!(cudaStreamWaitEvent(w.policy_s, w.tower_e, 0));
        check!(cudaStreamWaitEvent(w.value_s, w.tower_e, 0));

        // policy head (21p_policy)
        check!(cudnnSetStream(w.handle_dnn, w.policy_s));
        check!(cublasSetStream_v2(w.handle_blas, w.policy_s));
        check!(cudnnConvolutionForward(
            w.handle_dnn,
            &c_1,  // alpha
            w.residual_t, w.residual_1,  // input
            w.network.policy_f, w.network.weights["21p_policy/downsample:0"],  // weights
            w.network.conv2d_1,  // convolution
            ConvolutionFwdAlgo::ImplicitPrecompGemm,  // algo
            w.scratch_1, w.scratch_size,  // workspace
            &c_0,  // beta
            w.policy_t, w.policy_1,  // output
        ), "21p_policy/down", w.policy_1, w.batch_size, 2, 361);

        check!(cudnnBatchNormalizationForwardInference(
            w.handle_dnn,
            BatchNormMode::Spatial,
            &c_1,  // alpha
            &c_0,  // beta
            w.policy_t, w.policy_1,  // input
            w.policy_t, w.policy_2,  // output
            w.policy_bn_t,
            w.network.ones, w.network.zeros,  // scale, bias
            w.network.weights["21p_policy/mean:0"],
            w.network.weights["21p_policy/variance:0"],
            epsilon
        ), "21p_policy/down_bn", w.policy_2, w.batch_size, 2, 361);

        check!(cudnnActivationForward(
            w.handle_dnn,
            w.network.relu,
            &c_1,  // alpha
            w.policy_t, w.policy_2,  // input
            &c_0,  // beta
            w.policy_t, w.policy_2,  // output
        ), "21p_policy/down_relu", w.policy_2, w.batch_size, 2, 361);

        check!(cublasSgemm_v2(
            w.handle_blas,
            Operation::N,
            Operation::N,
            362, w.batch_size as i32, 722, // output_dims, batch_size, input_dims
            &c_1,  // alpha
            w.network.weights["21p_policy/weights:0"], 362,  // input_2
            w.policy_2, 722,  // input_1
            &c_0,  // beta
            w.policy_1, 362  // output
        ), "21p_policy/ff", w.policy_1, w.batch_size, 362);

        check!(cudnnAddTensor(
            w.handle_dnn,
            &c_1,  // alpha
            w.network.policy_bias_t, w.network.weights["21p_policy/bias:0"],  // bias
            &c_1,  // beta
            w.policy_softmax_t, w.policy_1  // input and output
        ), "21p_policy/bias", w.policy_1, w.batch_size, 362);

        check!(cudnnSoftmaxForward(
            w.handle_dnn,
            SoftmaxAlgorithm::Accurate,
            SoftmaxMode::Instance,
            &c_1,  // alpha
            w.policy_softmax_t, w.policy_1,  // input
            &c_0,  // beta
            w.policy_softmax_t, w.policy_2  // output
        ), "21p_policy/softmax", w.policy_2, w.batch_size, 362);

        for i in 0..w.batch_size {
            check!(cudaMemcpyAsync(
                softmax[i].as_mut_ptr() as *mut c_void,
                w.policy_2.offset((i * 1448) as isize),
                1448,
                MemcpyKind::DeviceToHost,
                w.policy_s
            ));
        }

        // value head (21v_value)
        check!(cudnnSetStream(w.handle_dnn, w.value_s));
        check!(cublasSetStream_v2(w.handle_blas, w.value_s));
        check!(cudnnConvolutionForward(
            w.handle_dnn,
            &c_1,  // alpha
            w.residual_t, w.residual_1,  // input
            w.network.value_f, w.network.weights["21v_value/downsample:0"],  // weights
            w.network.conv2d_1,  // convolution
            ConvolutionFwdAlgo::ImplicitPrecompGemm,  // algo
            w.scratch_2, w.scratch_size,  // workspace
            &c_0,  // beta
            w.value_t, w.value_1  // output
        ), "21v_value/down", w.value_1, w.batch_size, 361);

        check!(cudnnBatchNormalizationForwardInference(
            w.handle_dnn,
            BatchNormMode::Spatial,
            &c_1,  // alpha
            &c_0,  // beta
            w.value_t, w.value_1,  // input
            w.value_t, w.value_2,  // output
            w.value_bn_t,
            w.network.ones, w.network.zeros,  // scale, bias
            w.network.weights["21v_value/mean:0"],
            w.network.weights["21v_value/variance:0"],
            epsilon
        ), "21v_value/down_bn", w.value_2, w.batch_size, 361);

        check!(cudnnActivationForward(
            w.handle_dnn,
            w.network.relu,
            &c_1,  // alpha
            w.value_t, w.value_2,  // input
            &c_0,  // beta
            w.value_t, w.value_2,  // output
        ), "21v_value/down_relu", w.value_2, w.batch_size, 361);

        check!(cublasSgemm_v2(
            w.handle_blas,
            Operation::N,
            Operation::N,
            256, w.batch_size as i32, 361,  // output_dims, batch_size, input_dims
            &c_1,  // alpha
            w.network.weights["21v_value/weights_1:0"], 256,  // input_2
            w.value_2, 361,  // input_1
            &c_0,  // beta
            w.value_1, 256  // output
        ), "21v_value/ff_256", w.value_1, w.batch_size, 256);

        check!(cudnnAddTensor(
            w.handle_dnn,
            &c_1,  // alpha
            w.network.value_256_bias_t, w.network.weights["21v_value/bias_1:0"],  // bias
            &c_1,  // beta
            w.value_256_t, w.value_1  // input and output
        ), "21v_value/ff_bias_256", w.value_1, w.batch_size, 256);

        check!(cudnnActivationForward(
            w.handle_dnn,
            w.network.relu,
            &c_1,  // alpha
            w.value_256_t, w.value_1,  // input
            &c_0,  // beta
            w.value_256_t, w.value_1,  // output
        ), "21v_value/ff_relu_256", w.value_1, w.batch_size, 256);

        check!(cublasSgemm_v2(
            w.handle_blas,
            Operation::N,
            Operation::N,
            1, w.batch_size as i32, 256,  // output_dims, batch_size, input_dims
            &c_1,  // alpha
            w.network.weights["21v_value/weights_2:0"], 1,  // input_2
            w.value_1, 256,  // input_1
            &c_0,  // beta
            w.value_2, 1  // output
        ), "21v_value/ff_1", w.value_2, w.batch_size, 1);

        check!(cudnnAddTensor(
            w.handle_dnn,
            &c_1,  // alpha
            w.network.value_1_bias_t, w.network.weights["21v_value/bias_2:0"],  // bias
            &c_1,  // beta
            w.value_1_t, w.value_2  // input and output
        ), "21v_value/ff_bias_1", w.value_2, w.batch_size, 1);

        check!(cudnnActivationForward(
            w.handle_dnn,
            w.network.tanh,
            &c_1,  // alpha
            w.value_1_t, w.value_2,  // input
            &c_0,  // beta
            w.value_1_t, w.value_2,  // output
        ), "21v_value/ff_tanh_2", w.value_2, w.batch_size, 1);

        check!(cudaMemcpyAsync(
            value.as_mut_ptr() as *mut c_void,
            w.value_2,
            4 * w.batch_size,
            MemcpyKind::DeviceToHost,
            w.value_s
        ));

        // wait for both the value and policy head to finish
        check!(cudaStreamSynchronize(w.policy_s));
        check!(cudaStreamSynchronize(w.value_s));
    }

    (value, softmax.into_iter().map(|s| s.into_boxed_slice()).collect())
}
