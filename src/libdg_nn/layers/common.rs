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

use std::collections::HashMap;

use dg_cuda::cudnn;

use crate::tensor::Tensor;

/// The number of channels to assume if not given in the network weights file.
pub const DEFAULT_NUM_CHANNELS: i32 = 128;

/// The number of samples to assume if not given in the network weights file.
pub const DEFAULT_NUM_SAMPLES: i32 = 8;

/// Returns the number of channels to in each layer of the graph.
/// 
/// # Arguments
/// 
/// * `tensors` -
/// 
pub fn get_num_channels(tensors: &HashMap<String, Tensor>) -> i32 {
    tensors.get("num_channels:0")
        .map(|x| { x.as_i32() })
        .unwrap_or(DEFAULT_NUM_CHANNELS)
}

/// Returns the number of samples to use internally in the policy & value
/// head.
/// 
/// # Arguments
/// 
/// * `tensors` -
/// 
pub fn get_num_samples(tensors: &HashMap<String, Tensor>) -> i32 {
    tensors.get("num_samples:0")
        .map(|x| { x.as_i32() })
        .unwrap_or(DEFAULT_NUM_SAMPLES)
}

/// Returns a `TensorDescriptor` for an feature tensor for the given
/// `batch_size` and `num_channels`.
/// 
/// # Arguments
/// 
/// * `batch_size` -
/// * `num_channels` -
/// * `width_height` -
/// 
pub fn create_tensor_descriptor(batch_size: i32, num_channels: i32, width_height: i32) -> Result<cudnn::TensorDescriptor, cudnn::Status> {
    cudnn::TensorDescriptor::new(
        cudnn::TensorFormat::NHWC,
        cudnn::DataType::Half,
        &[batch_size, num_channels, width_height, width_height]
    )
}

/// Returns a `TensorDescriptor` for an offset tensor for the given
/// `num_channels`.
/// 
/// # Arguments
/// 
/// * `num_channels` -
/// 
pub fn create_offset_descriptor(num_channels: i32) -> Result<cudnn::TensorDescriptor, cudnn::Status> {
    cudnn::TensorDescriptor::new(
        cudnn::TensorFormat::NHWC,
        cudnn::DataType::Half,
        &[1, num_channels, 1, 1]
    )
}

/// Returns a `TensorDescriptor` for a dense tensor for the given `batch_size`
/// and `size`.
///
/// # Arguments
/// 
/// * `batch_size` -
/// * `size` -
/// 
pub fn create_dense_descriptor(batch_size: i32, size: i32) -> Result<cudnn::TensorDescriptor, cudnn::Status> {
    cudnn::TensorDescriptor::new(
        cudnn::TensorFormat::NHWC,
        cudnn::DataType::Half,
        &[batch_size, size, 1, 1]
    )
}