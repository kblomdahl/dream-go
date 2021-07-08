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

use crate::layers::{Conv2d, create_offset_descriptor, get_num_channels};
use crate::tensor::Tensor;
use crate::Error;

pub struct ResidualLayer {
    conv_1: Conv2d,
    conv_2: Conv2d,
    scale_offset: cudnn::Scale
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
    pub fn new(
        handle: &cudnn::Handle,
        n: i32,
        i: usize,
        tensors: &HashMap<String, Tensor>
    ) -> Result<Option<ResidualLayer>, Error>
    {
        let weights_1 = tensors.get(&format!("{:02}_residual/conv_1:0", i));
        let weights_2 = tensors.get(&format!("{:02}_residual/conv_2:0", i));
        let alpha = tensors.get(&format!("{:02}_residual/alpha:0", i));

        if weights_1.is_none() || weights_2.is_none() {
            return Ok(None);
        }

        let num_channels = get_num_channels(tensors);
        let gate_t = alpha.map(|t| t.as_f32()).unwrap_or(0.5);

        Ok(Some(ResidualLayer {
            conv_1: Conv2d::new(n, [num_channels, num_channels, 3, 3])
                        .with_tensors(tensors, &format!("{:02}_residual/conv_1", i))
                        .build(handle)?,
            conv_2: Conv2d::new(n, [num_channels, num_channels, 3, 3])
                        .with_alpha([gate_t, 1.0 - gate_t])
                        .with_tensors(tensors, &format!("{:02}_residual/conv_2", i))
                        .build(handle)?,
            scale_offset: cudnn::Scale::new(create_offset_descriptor(num_channels)?, gate_t)?
        }))
    }

    pub fn forward<'a, A: cuda::Allocator + Clone>(
        &self,
        handle: &cudnn::Handle,
        input: cuda::SmartPtr<A>,
        allocator: &mut A,
        stream: &cuda::Stream,
    ) -> Result<cuda::SmartPtr<A>, Error>
    {
        if self.conv_2.prepare(handle, stream)? {
            self.scale_offset.forward(&handle, self.conv_2.offset().get().as_ptr())?;
        }

        let y = self.conv_1.forward(&handle, &input, allocator, stream)?;
        self.conv_2.forward_skip(&handle, &y, &input, allocator, stream)
    }
}

#[cfg(test)]
mod tests {
    use test::{Bencher, black_box};
    use dg_cuda::cudnn::{DataType, Handle};
    use dg_cuda::{Native, malloc, Stream};
    use dg_utils::types::f16;
    use super::*;

    #[bench]
    fn residual_block(b: &mut Bencher) {
        let mut tensors = HashMap::new();
        tensors.insert("01_residual/conv_1:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(1.0); 3 * 3 * 128 * 128]).unwrap());
        tensors.insert("01_residual/conv_1/offset:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(0.5); 128]).unwrap());
        tensors.insert("01_residual/conv_2:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(1.0); 3 * 3 * 128 * 128]).unwrap());
        tensors.insert("01_residual/conv_2/offset:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(0.5); 128]).unwrap());
        tensors.insert("01_residual/alpha:0".to_string(), Tensor::from_vec(DataType::Float, vec! [0.5f32; 1]).unwrap());

        let handle = Handle::new().expect("could not create cudnn handle");
        let batch_size = 16;
        let residual_block = ResidualLayer::new(&handle, batch_size, 1, &tensors).expect("could not create residual layer").expect("could not find weights");
        let mut allocator = Native::new();
        let stream = Stream::default();

        b.iter(move || {
            let x = malloc(batch_size as usize * 128 * 19 * 19, &allocator).expect("could not allocate input buffer");
            let y = residual_block.forward(&handle, black_box(x), &mut allocator, &stream);

            assert!(y.is_ok());
            y
        })
    }
}
