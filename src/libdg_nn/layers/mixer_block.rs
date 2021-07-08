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

use crate::layers::{Dense, get_num_channels};
use crate::tensor::Tensor;
use crate::Error;

#[allow(unused)]
pub struct MixerLayer {
    mlp_block_1_1: Dense,
    mlp_block_1_2: Dense,
    mlp_block_2_1: Dense,
    mlp_block_2_2: Dense
}

impl MixerLayer {
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
    #[allow(unused)]
    pub fn new(
        handle: &cudnn::Handle,
        batch_size: i32,
        i: usize,
        tensors: &HashMap<String, Tensor>
    ) -> Result<Option<Self>, Error>
    {
        let token_mlp_dims = 361;
        let channels_mlp_dims = get_num_channels(tensors);

        Ok(Some(Self {
            mlp_block_1_1: Dense::new(batch_size, [token_mlp_dims, token_mlp_dims])
                .with_input_strides([token_mlp_dims * channels_mlp_dims, 19, 1, channels_mlp_dims])
                .with_tensors(tensors, &format!("{:02}_mixer/mlp_1/linear_1", i))
                .build(handle)?,
            mlp_block_1_2: Dense::new(batch_size, [token_mlp_dims, token_mlp_dims])
                .with_skip_strides([token_mlp_dims * channels_mlp_dims, 19, 1, channels_mlp_dims])
                .with_tensors(tensors, &format!("{:02}_mixer/mlp_1/linear_2", i))
                .build(handle)?,
            mlp_block_2_1: Dense::new(batch_size, [channels_mlp_dims, channels_mlp_dims])
                .with_input_strides([token_mlp_dims * channels_mlp_dims, 19, 1, token_mlp_dims])
                .with_tensors(tensors, &format!("{:02}_mixer/mlp_2/linear_1", i))
                .build(handle)?,
            mlp_block_2_2: Dense::new(batch_size, [channels_mlp_dims, channels_mlp_dims])
                .with_output_strides([token_mlp_dims * channels_mlp_dims, 19, 1, token_mlp_dims])
                .with_tensors(tensors, &format!("{:02}_mixer/mlp_2/linear_2", i))
                .build(handle)?,
        }))
    }

    #[allow(unused)]
    pub fn forward<'a, A: cuda::Allocator + Clone>(
        &self,
        handle: &cudnn::Handle,
        input: cuda::SmartPtr<A>,
        allocator: &mut A,
        stream: &cuda::Stream,
    ) -> Result<cuda::SmartPtr<A>, Error>
    {
        let y = self.mlp_block_1_1.forward(handle, &input, allocator, stream)?;
        let y = self.mlp_block_1_2.forward_skip(handle, &y, &input, allocator, stream)?;
        let y = self.mlp_block_2_1.forward(handle, &y, allocator, stream)?;
        self.mlp_block_2_2.forward_skip(handle, &y, &input, allocator, stream)
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
    fn mixer_block(b: &mut Bencher) {
        let mut tensors = HashMap::new();
        tensors.insert("01_mixer/mlp_1/linear_1:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(1.0); 361 * 361]).unwrap());
        tensors.insert("01_mixer/mlp_1/linear_1/offset:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(0.5); 361]).unwrap());
        tensors.insert("01_mixer/mlp_1/linear_2:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(1.0); 361 * 361]).unwrap());
        tensors.insert("01_mixer/mlp_1/linear_2/offset:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(0.5); 361]).unwrap());
        tensors.insert("01_mixer/mlp_2/linear_1:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(1.0); 128 * 128]).unwrap());
        tensors.insert("01_mixer/mlp_2/linear_1/offset:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(0.5); 128]).unwrap());
        tensors.insert("01_mixer/mlp_2/linear_2:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(1.0); 128 * 128]).unwrap());
        tensors.insert("01_mixer/mlp_2/linear_2/offset:0".to_string(), Tensor::from_vec(DataType::Half, vec! [f16::from(0.5); 128]).unwrap());

        let handle = Handle::new().expect("could not create cudnn handle");
        let batch_size = 16;
        let mixer_block = MixerLayer::new(&handle, batch_size, 1, &tensors).expect("could not create bottlenet layer").expect("could not find weights");
        let mut allocator = Native::new();
        let stream = Stream::default();

        b.iter(move || {
            let x = malloc(batch_size as usize * 128 * 19 * 19, &allocator).expect("could not allocate input buffer");
            let y = mixer_block.forward(&handle, black_box(x), &mut allocator, &stream);

            assert!(y.is_ok());
            y
        })
    }
}
