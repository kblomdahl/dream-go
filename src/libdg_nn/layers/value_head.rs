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
use crate::layers::{Conv2d, Dense, create_dense_descriptor, get_num_channels};
use crate::Error;

pub struct ValueLayer {
    conv_1: Conv2d,
    linear_2: Dense,
    tanh: cudnn::Activation
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
        let num_samples = 2;

        Ok(ValueLayer {
            conv_1: Conv2d::new(n, [num_samples, num_channels, 3, 3])
                        .with_activation(cudnn::ActivationDescriptor::relu()?)
                        .with_compute_type(cudnn::DataType::Float)
                        .with_tensors(tensors, &format!("{:02}v_value/conv_1", i))
                        .build(handle)?,
            linear_2: Dense::new(n, [1, 361*num_samples])
                        .with_activation(cudnn::ActivationDescriptor::identity()?)
                        .with_tensors(tensors, &format!("{:02}v_value/linear_2", i))
                        .build(handle)?,
            tanh: Self::create_tanh_activation(n)?
        })
    }

    fn create_tanh_activation(n: i32) -> Result<cudnn::Activation, cudnn::Status> {
        cudnn::Activation::new(
            cudnn::ActivationDescriptor::tanh()?,
            create_dense_descriptor(n, 1)?,
            create_dense_descriptor(n, 1)?,
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

        // perform the linear feed-forward layer without the final activation
        // in the convolution since it's bugged :'(
        let value_2 = self.linear_2.forward(handle, &value_1, allocator, stream)?;
        self.tanh.forward(handle, value_2.as_ptr(), value_2.as_ptr())?;

        Ok(value_2)
    }
}
