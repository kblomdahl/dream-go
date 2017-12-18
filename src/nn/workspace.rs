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
use std::sync::Arc;

use nn::ffi::*;
use nn::ffi::cublas::*;
use nn::ffi::cuda::*;
use nn::ffi::cudnn::*;
use nn::loader;

/// The width of each filter in the neural network. Larger is "better" but takes
/// longer to train and gives worse runtime performance during inference.
pub const NUM_FEATURES: usize = 128;

/// Read-only data necessary to compute the neural network that can be shared
/// between multiple workspaces without being copied.
pub struct Shared {
    pub(super) handle_blas: cublas::Handle,
    pub(super) handle_dnn: cudnn::Handle,

    /// The data-type to use during computation. This will be either FLOAT or HALF
    /// depending on the _compute capability_ of the given device.
    pub(super) data_type: DataType,

    // convolutional descriptors
    pub(super) conv2d_1: ConvolutionDescriptor,
    pub(super) conv2d_3: ConvolutionDescriptor,

    // activation functions
    pub(super) relu: ActivationDescriptor,
    pub(super) tanh: ActivationDescriptor,

    // filter dimensions
    pub(super) up_f: FilterDescriptor,
    pub(super) residual_f: FilterDescriptor,
    pub(super) value_f: FilterDescriptor,
    pub(super) policy_f: FilterDescriptor,

    // bias tensors
    pub(super) policy_bias_t: TensorDescriptor,
    pub(super) value_256_bias_t: TensorDescriptor,
    pub(super) value_1_bias_t: TensorDescriptor,

    /// The weights of this network
    pub(super) weights: HashMap<String, *const c_void>,
    pub(super) zeros: *mut c_void,
    pub(super) ones: *mut c_void
}

impl Shared {
    pub fn new(path: &Path, data_type: DataType) -> Option<Shared> {
        assert!(data_type == DataType::Float || data_type == DataType::Half);

        if let Some(weights) = loader::load(path, data_type) {
            let mut n = Shared {
                handle_blas: ptr::null_mut(),
                handle_dnn: ptr::null_mut(),

                data_type: data_type,

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
                    n.data_type
                ));

                check!(cudnnCreateConvolutionDescriptor(&mut n.conv2d_3));
                check!(cudnnSetConvolution2dDescriptor(
                    n.conv2d_3,
                    1, 1,
                    1, 1,
                    1, 1,
                    ConvolutionMode::CrossCorrelation,
                    n.data_type
                ));

                #[cfg(feature = "tensor-core")]
                {
                    check!(cudnnSetConvolutionMathType(n.conv2d_1, MathType::TensorOpMath));
                    check!(cudnnSetConvolutionMathType(n.conv2d_3, MathType::TensorOpMath));
                }

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
                    n.data_type,
                    TensorFormat::NCHW,
                    NUM_FEATURES as i32, 34, 3, 3
                ));

                check!(cudnnCreateFilterDescriptor(&mut n.residual_f));
                check!(cudnnSetFilter4dDescriptor(
                    n.residual_f,
                    n.data_type,
                    TensorFormat::NCHW,
                    NUM_FEATURES as i32, NUM_FEATURES as i32, 3, 3
                ));

                check!(cudnnCreateFilterDescriptor(&mut n.value_f));
                check!(cudnnSetFilter4dDescriptor(
                    n.value_f,
                    n.data_type,
                    TensorFormat::NCHW,
                    1, NUM_FEATURES as i32, 1, 1
                ));

                check!(cudnnCreateFilterDescriptor(&mut n.policy_f));
                check!(cudnnSetFilter4dDescriptor(
                    n.policy_f,
                    n.data_type,
                    TensorFormat::NCHW,
                    2, NUM_FEATURES as i32, 1, 1
                ));

                check!(cudnnCreateTensorDescriptor(&mut n.policy_bias_t));
                check!(cudnnSetTensor4dDescriptor(
                    n.policy_bias_t,
                    TensorFormat::NCHW,
                    n.data_type,
                    1, 362, 1, 1
                ));

                check!(cudnnCreateTensorDescriptor(&mut n.value_256_bias_t));
                check!(cudnnSetTensor4dDescriptor(
                    n.value_256_bias_t,
                    TensorFormat::NCHW,
                    n.data_type,
                    1, 256, 1, 1
                ));

                check!(cudnnCreateTensorDescriptor(&mut n.value_1_bias_t));
                check!(cudnnSetTensor4dDescriptor(
                    n.value_1_bias_t,
                    TensorFormat::NCHW,
                    n.data_type,
                    1, 1, 1, 1
                ));

                // The `.zeros` and `.ones` are always f32 regardless of the _main_ data
                // type since they are only used for batch-normalization which requires
                // all of the `scale`, `bias`, `estimated_mean`, and `estimated_variance`
                // values to be `f32`.
                check!(cudaMalloc(&mut n.zeros, NUM_FEATURES * DataType::Float.size()));
                check!(cudaMalloc(&mut n.ones, NUM_FEATURES * DataType::Float.size()));

                check!(cudaMemcpy(
                    n.zeros,
                    vec! [0.0f32; NUM_FEATURES].as_ptr() as *const c_void,
                    NUM_FEATURES * DataType::Float.size(),
                    MemcpyKind::HostToDevice
                ));

                check!(cudaMemcpy(
                    n.ones,
                    vec! [1.0f32; NUM_FEATURES].as_ptr() as *const c_void,
                    NUM_FEATURES * DataType::Float.size(),
                    MemcpyKind::HostToDevice
                ));
            }

            Some(n)
        } else {
            None
        }
    }

    /// Returns whether this network is running is half precision mode.
    pub fn is_half(&self) -> bool {
        return self.data_type == DataType::Half;
    }

    /// Returns the best convolution algorithm for the given filter size.
    /// 
    /// # Arguments
    /// 
    /// * `filter_size` - The width and height of the filter
    /// 
    pub fn get_convolution_algo(&self, filter_size: usize) -> ConvolutionFwdAlgo {
        if self.is_half() {
            match filter_size {
                3 => ConvolutionFwdAlgo::WinogradNonFused,
                _ => ConvolutionFwdAlgo::ImplicitPrecompGemm
            }
        } else {
            match filter_size {
                3 => ConvolutionFwdAlgo::Winograd,
                _ => ConvolutionFwdAlgo::ImplicitPrecompGemm
            }
        }
    }
}

impl Drop for Shared {
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

/// A per-thread workspace containing tensor descriptors, operations, and
/// data arrays necessary to perform a forward pass through the neural
/// network.
pub struct Workspace {
    pub(super) shared: Arc<Shared>,
    pub(super) handle_blas: cublas::Handle,
    pub(super) handle_dnn: cudnn::Handle,
    pub(super) batch_size: usize,

    pub(super) tower_s: Stream,
    pub(super) policy_s: Stream,
    pub(super) value_s: Stream,
    pub(super) tower_e: Event,

    pub(super) input_t: TensorDescriptor,
    pub(super) residual_t: TensorDescriptor,
    pub(super) residual_bn_t: TensorDescriptor,
    pub(super) value_t: TensorDescriptor,
    pub(super) value_bn_t: TensorDescriptor,
    pub(super) value_256_t: TensorDescriptor,
    pub(super) value_1_t: TensorDescriptor,
    pub(super) policy_t: TensorDescriptor,
    pub(super) policy_bn_t: TensorDescriptor,
    pub(super) policy_softmax_t: TensorDescriptor,

    pub(super) input: *mut c_void,
    pub(super) residual_1: *mut c_void,
    pub(super) residual_2: *mut c_void,
    pub(super) residual_3: *mut c_void,
    pub(super) policy_1: *mut c_void,
    pub(super) policy_2: *mut c_void,
    pub(super) value_1: *mut c_void,
    pub(super) value_2: *mut c_void,

    pub(super) scratch_size: usize,
    pub(super) scratch_1: *mut c_void,
    pub(super) scratch_2: *mut c_void
}

impl Workspace {
    /// Returns a structure containing the mutable workspace necessary
    /// to perform a forward pass through the network with the given
    /// batch size.
    /// 
    /// # Arguments
    /// 
    /// * `network` -
    /// * `batch_size` -
    /// 
    pub fn new(shared: &Arc<Shared>, batch_size: usize) -> Workspace {
        assert!(batch_size >= 1);

        let mut w = Workspace {
            shared: shared.clone(),
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
                shared.data_type,
                batch_size as i32, 34, 19, 19
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.residual_t));
            check!(cudnnSetTensor4dDescriptor(
                w.residual_t,
                TensorFormat::NCHW,
                shared.data_type,
                batch_size as i32, NUM_FEATURES as i32, 19, 19
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.residual_bn_t));
            check!(cudnnSetTensor4dDescriptor(
                w.residual_bn_t,
                TensorFormat::NCHW,
                DataType::Float,  // batch normalization parameters are always `f32`
                1, NUM_FEATURES as i32, 1, 1
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.value_t));
            check!(cudnnSetTensor4dDescriptor(
                w.value_t,
                TensorFormat::NCHW,
                shared.data_type,
                batch_size as i32, 1, 19, 19
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.value_bn_t));
            check!(cudnnSetTensor4dDescriptor(
                w.value_bn_t,
                TensorFormat::NCHW,
                DataType::Float,  // batch normalization parameters are always `f32`
                1, 1, 1, 1
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.value_256_t));
            check!(cudnnSetTensor4dDescriptor(
                w.value_256_t,
                TensorFormat::NCHW,
                shared.data_type,
                batch_size as i32, 256, 1, 1
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.value_1_t));
            check!(cudnnSetTensor4dDescriptor(
                w.value_1_t,
                TensorFormat::NCHW,
                shared.data_type,
                batch_size as i32, 1, 1, 1
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.policy_t));
            check!(cudnnSetTensor4dDescriptor(
                w.policy_t,
                TensorFormat::NCHW,
                shared.data_type,
                batch_size as i32, 2, 19, 19
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.policy_bn_t));
            check!(cudnnSetTensor4dDescriptor(
                w.policy_bn_t,
                TensorFormat::NCHW,
                DataType::Float,  // batch normalization parameters are always `f32`
                1, 2, 1, 1
            ));

            check!(cudnnCreateTensorDescriptor(&mut w.policy_softmax_t));
            check!(cudnnSetTensor4dDescriptor(
                w.policy_softmax_t,
                TensorFormat::NCHW,
                shared.data_type,
                batch_size as i32, 362, 1, 1
            ));

            check!(cudaMalloc(&mut w.input, batch_size * shared.data_type.size() * 12274));
            check!(cudaMalloc(&mut w.residual_1, batch_size * shared.data_type.size() * 92416));
            check!(cudaMalloc(&mut w.residual_2, batch_size * shared.data_type.size() * 92416));
            check!(cudaMalloc(&mut w.residual_3, batch_size * shared.data_type.size() * 92416));
            check!(cudaMalloc(&mut w.policy_1, batch_size * shared.data_type.size() * 722));
            check!(cudaMalloc(&mut w.policy_2, batch_size * shared.data_type.size() * 722));
            check!(cudaMalloc(&mut w.value_1, batch_size * shared.data_type.size() * 361));
            check!(cudaMalloc(&mut w.value_2, batch_size * shared.data_type.size() * 361));

            // allocate two scratch workspaces that are at least as large as the
            // largest workspace requested by cuDNN.
            let mut up_s: usize = 0;
            let mut residual_s: usize = 0;
            let mut value_s: usize = 0;
            let mut policy_s: usize = 0;

            check!(cudnnGetConvolutionForwardWorkspaceSize(
                shared.handle_dnn,
                w.input_t,
                shared.up_f,
                shared.conv2d_3,
                w.residual_t,
                shared.get_convolution_algo(3),
                &mut up_s
            ));

            check!(cudnnGetConvolutionForwardWorkspaceSize(
                shared.handle_dnn,
                w.residual_t,
                shared.residual_f,
                shared.conv2d_3,
                w.residual_t,
                shared.get_convolution_algo(3),
                &mut residual_s
            ));

            check!(cudnnGetConvolutionForwardWorkspaceSize(
                shared.handle_dnn,
                w.residual_t,
                shared.value_f,
                shared.conv2d_1,
                w.value_t,
                shared.get_convolution_algo(1),
                &mut value_s
            ));

            check!(cudnnGetConvolutionForwardWorkspaceSize(
                shared.handle_dnn,
                w.residual_t,
                shared.policy_f,
                shared.conv2d_1,
                w.policy_t,
                shared.get_convolution_algo(1),
                &mut policy_s
            ));

            w.scratch_size = vec! [up_s, residual_s, value_s, policy_s].into_iter().max().unwrap();

            check!(cudaMalloc(&mut w.scratch_1, w.scratch_size));
            check!(cudaMalloc(&mut w.scratch_2, w.scratch_size));
        }

        w
    }
}

impl Drop for Workspace {
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
