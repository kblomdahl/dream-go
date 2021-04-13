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
use dg_go::utils::features;
use std::collections::HashMap;

use crate::layers::{Conv2d, get_num_channels};
use crate::tensor::Tensor;
use crate::Error;

pub struct UpLayer {
    up: Conv2d
}

impl UpLayer {
    /// Create a single convolutional layer.
    ///
    /// # Arguments
    ///
    /// * `handle` - The cuDNN handle
    /// * `n` - The number of images.
    /// * `tensors` -
    ///
    pub fn new(handle: &cudnn::Handle, n: i32, tensors: &HashMap<String, Tensor>) -> Result<UpLayer, Error> {
        let num_features = features::Default::num_features();
        let num_channels = get_num_channels(tensors);

        Ok(UpLayer {
            up: Conv2d::new(n, [num_channels, num_features as i32, 3, 3])
                    .with_tensors(tensors, "01_upsample/conv_1")
                    .build(handle)?
        })
    }

    pub fn forward<'a, A: cuda::Allocator + Clone>(
        &self,
        handle: &cudnn::Handle,
        input: &cuda::SmartPtr<A>,
        allocator: &mut A,
        stream: &cuda::Stream
    ) -> Result<cuda::SmartPtr<A>, Error>
    {
        self.up.forward(handle, input, allocator, stream)
    }
}
