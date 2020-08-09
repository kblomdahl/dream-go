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

use dg_cuda::cudnn as cudnn2;
use dg_cuda as cuda2;
use std::collections::HashMap;

use crate::graph::{DEFAULT_NUM_CHANNELS, InferenceType, Conv2d, create_offset_descriptor, create_convolution_bias_activation_3x3};
use crate::tensor::Tensor;
use crate::Error;

pub struct ResidualLayer {
    conv_1: Conv2d,
    conv_2: Conv2d,
    scale_offset: cudnn2::Scale
}

impl ResidualLayer {
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
    pub unsafe fn new(handle: &cudnn2::Handle, n: i32, i: usize, tensors: &HashMap<String, Tensor>) -> Result<Option<ResidualLayer>, Error> {
        let weights_1 = tensors.get(&format!("{:02}_residual/conv_1:0", i));
        let weights_2 = tensors.get(&format!("{:02}_residual/conv_2:0", i));
        let alpha = tensors.get(&format!("{:02}_residual/alpha:0", i));

        if weights_1.is_none() || weights_2.is_none() {
            return Ok(None);
        }

        let num_channels = tensors.get("num_channels:0")
            .map(|x| { x.as_i32() })
            .unwrap_or(DEFAULT_NUM_CHANNELS);
        let gate_t = alpha.map(|t| t.as_f32()).unwrap_or(0.5);

        Ok(Some(ResidualLayer {
            conv_1: Conv2d::new(format!("{:02}_residual/conv_1", i), create_convolution_bias_activation_3x3(handle, n, num_channels, num_channels, &[1.0, 0.0])?, tensors)?,
            conv_2: Conv2d::new(format!("{:02}_residual/conv_2", i), create_convolution_bias_activation_3x3(handle, n, num_channels, num_channels, &[gate_t, 1.0 - gate_t])?, tensors)?,
            scale_offset: cudnn2::Scale::new(create_offset_descriptor(num_channels)?, gate_t)?
        }))
    }

    pub unsafe fn forward<'a, A: cuda2::Allocator + Clone, T: InferenceType>(
        &self,
        handle: &cudnn2::Handle,
        input: cuda2::SmartPtr<A>,
        allocator: &mut A,
        stream: &cuda2::Stream,
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        if self.conv_2.prepare(handle, stream)? {
            self.scale_offset.forward(&handle, self.conv_2.offset.get().as_ptr())?;
        }

        let y = self.conv_1.forward(&handle, &input, allocator, stream)?;
        self.conv_2.forward_skip(&handle, &y, &input, allocator, stream)
    }
}
