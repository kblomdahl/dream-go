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

use dg_cuda::cudnn;
use dg_cuda as cuda;
use std::collections::HashMap;

use crate::tensor::Tensor;
use crate::layers::{Conv2d, GlobalPooling, create_dense_descriptor, get_num_channels};
use crate::Error;

pub struct ValueLayer {
    conv_1: Conv2d,
    reduce_mean: GlobalPooling,
    tanh: cudnn::Activation,
}

impl ValueLayer {
    /// Create a layer that takes the final output of the residual block and
    /// transforms it into a scalar value.
    ///
    /// # Arguments
    ///
    /// * `handle` - The cuDNN handle
    /// * `n` - The number of images.
    /// * `i` - The index of the layer.
    /// * `tensors` -
    ///
    pub fn new(handle: &cudnn::Handle, n: i32, i: usize, tensors: &HashMap<String, Tensor>) -> Result<ValueLayer, Error> {
        let num_channels = get_num_channels(tensors);

        Ok(ValueLayer {
            conv_1: Conv2d::new(n, [8, num_channels, 3, 3])
                        .with_activation(cudnn::ActivationDescriptor::identity()?)
                        .with_tensors(tensors, &format!("{:02}v_value/conv_1", i))
                        .build(handle)?,
            reduce_mean: GlobalPooling::new(n, 8, cudnn::ReduceTensorOp::Avg)
                        .build()?,
            tanh: Self::create_dense_tanh(n, 1)?
        })
    }

    /// Returns an `Activation` structure that performs an element-wise tanh
    /// activation on the entire tensor.
    /// 
    /// # Arguments
    /// 
    /// * `batch_size` - 
    /// * `num_elements` - 
    /// 
    fn create_dense_tanh(
        batch_size: i32,
        num_elements: i32,
    ) -> Result<cudnn::Activation, cudnn::Status> {
        cudnn::Activation::new(
            cudnn::ActivationDescriptor::tanh()?, 
            create_dense_descriptor(batch_size, num_elements)?,
            create_dense_descriptor(batch_size, num_elements)?,
            [1.0, 0.0]
        )
    }

    pub fn forward<'a, A: cuda::Allocator + Clone>(
        &self,
        handle: &cudnn::Handle,
        input: &cuda::SmartPtr<A>,
        allocator: &mut A,
        stream: &cuda::Stream
    ) -> Result<cuda::SmartPtr<A>, Error>
    {
        // perform the forward convolution
        let value_1 = self.conv_1.forward(handle, input, allocator, stream)?;

        // perform the global average pooling
        let value_2 = self.reduce_mean.forward(handle, &value_1, allocator, stream)?;

        // perform the feed-forward linear layer (tanh)
        self.tanh.forward(handle, value_2.as_ptr(), value_2.as_ptr())?;

        Ok(value_2)
    }
}
