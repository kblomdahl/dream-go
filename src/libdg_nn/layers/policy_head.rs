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

use dg_cuda::cudnn as cudnn2;
use dg_cuda as cuda2;
use dg_utils::config;

use crate::tensor::Tensor;
use crate::layers::{Conv2d, Dense, create_dense_descriptor, create_offset_descriptor, get_num_channels};
use crate::Error;

pub struct PolicyLayer {
    conv_1: Conv2d,
    linear_2: Dense,
    softmax: cudnn2::Softmax,
    scale_tau: cudnn2::Scale,
}

impl PolicyLayer {
    /// Create a layer that takes the final output of the residual block and
    /// transforms it into a policy vector.
    ///
    /// # Arguments
    ///
    /// * `handle` - The cuDNN handle
    /// * `n` - The number of images.
    /// * `i` - The index of the layer.
    /// * `tensors` -
    ///
    pub fn new(handle: &cudnn2::Handle, n: i32, i: usize, tensors: &HashMap<String, Tensor>) -> Result<PolicyLayer, Error> {
        let num_channels = get_num_channels(tensors);
        let num_samples = 8;
        let tau = 1.0 / *config::SOFTMAX_TEMPERATURE;

        Ok(PolicyLayer {
            conv_1: Conv2d::new(n, [num_samples, num_channels, 3, 3])
                        .with_tensors(tensors, &format!("{:02}p_policy/conv_1", i))
                        .build(handle)?,
            linear_2: Dense::new(n, [362, 361*num_samples])
                        .with_alpha([tau, 0.0])
                        .with_activation(cudnn2::ActivationDescriptor::identity()?)
                        .with_tensors(tensors, &format!("{:02}p_policy/linear_1", i))
                        .build(handle)?,
            softmax: Self::create_softmax(n, 362)?,
            scale_tau: cudnn2::Scale::new(create_offset_descriptor(362)?, tau)?,
        })
    }

    /// Returns a `Softmax` structure for the given `batch_size` and `num_channels`.
    /// 
    /// # Arguments
    /// 
    /// * `batch_size` -
    /// * `num_channels` -
    /// 
    fn create_softmax(batch_size: i32, num_channels: i32) -> Result<cudnn2::Softmax, cudnn2::Status> {
        cudnn2::Softmax::new(
            cudnn2::SoftmaxMode::Instance,
            create_dense_descriptor(batch_size, num_channels)?,
            create_dense_descriptor(batch_size, num_channels)?,
            &[1.0, 0.0]
        )
    }

    pub fn forward<'a, A: cuda2::Allocator + Clone>(
        &self,
        handle: &cudnn2::Handle,
        input: &cuda2::SmartPtr<A>,
        allocator: &mut A,
        stream: &cuda2::Stream
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        if self.linear_2.prepare(handle, allocator, stream)? {
            self.scale_tau.forward(handle, self.linear_2.offset().get().as_ptr())?;
        }

        // perform the forward convolution
        let policy_1 = self.conv_1.forward(handle, input, allocator, stream)?;

        // perform the feed-forward linear layers
        let policy_2 = self.linear_2.forward(handle, &policy_1, allocator, stream)?;

        // softmax activation
        let policy_3 = cuda2::malloc(policy_2.size_in_bytes(), allocator)?;

        self.softmax.forward(handle, policy_2.as_ptr(), policy_3.as_ptr())?;

        Ok(policy_3)
    }
}