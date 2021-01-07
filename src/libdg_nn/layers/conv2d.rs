
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
use crate::layers::{create_tensor_descriptor, create_offset_descriptor};
use crate::Error;

pub struct Conv2d {
    conv_desc: cudnn::ConvolutionBiasActivation,
    filter: Tensor,
    offset: Tensor
}

pub struct Conv2dBuilder {
    batch_size: i32,
    width_height: i32,
    filter_shape: [i32; 4],
    alpha: [f32; 2],
    act_desc: Option<cudnn::ActivationDescriptor>,
    compute_type: cudnn::DataType,
    filter: Option<Tensor>,
    offset: Option<Tensor>,
}

impl Conv2dBuilder {
    fn new(
        batch_size: i32,
        filter: [i32; 4],
    ) -> Self
    {
        assert_eq!(filter[2], filter[3]);

        Self {
            batch_size: batch_size,
            width_height: 19,
            filter_shape: filter,
            alpha: [1.0, 0.0],
            act_desc: None,
            compute_type: if has_true_half() { cudnn::DataType::Half } else { cudnn::DataType::Float },
            filter: None,
            offset: None
        }
    }

    pub fn with_tensors(mut self, tensors: &HashMap<String, Tensor>, name: &str) -> Self {
        self.filter = Some(tensors.get(&format!("{}:0", name)).cloned().expect("no filter available"));
        self.offset = Some(tensors.get(&format!("{}/offset:0", name)).cloned().expect("no offset available"));
        self
    }

    pub fn with_compute_type(mut self, compute_type: cudnn::DataType) -> Self {
        self.compute_type = compute_type;
        self
    }

    #[cfg(test)]
    pub fn with_width_height(mut self, width_height: i32) -> Self {
        self.width_height = width_height;
        self
    }

    #[cfg(test)]
    pub fn with_filter(mut self, filter: Tensor) -> Self {
        self.filter = Some(filter);
        self
    }

    #[cfg(test)]
    pub fn with_offset(mut self, offset: Tensor) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn with_alpha(mut self, alpha: [f32; 2]) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_activation(mut self, act_desc: cudnn::ActivationDescriptor) -> Self {
        self.act_desc = Some(act_desc);
        self
    }

    fn create_convolution_descriptor(&self) -> Result<cudnn::ConvolutionDescriptor, cudnn::Status> {
        let filter_size = self.filter_shape[2];

        cudnn::ConvolutionDescriptor::new(
            &[filter_size / 2, filter_size / 2], // padding
            &[1, 1], // stride
            &[1, 1], // dilation
            cudnn::ConvolutionMode::CrossCorrelation,
            self.compute_type
        )
    }

    fn create_filter_descriptor(&self) -> Result<cudnn::FilterDescriptor, cudnn::Status> {
        cudnn::FilterDescriptor::new(
            cudnn::DataType::Half,
            cudnn::TensorFormat::NHWC,
            &self.filter_shape
        )
    }

    fn create_activation_descriptor(&mut self) -> Result<cudnn::ActivationDescriptor, cudnn::Status> {
        if let Some(act) = self.act_desc.take() {
            Ok(act)
        } else {
            cudnn::ActivationDescriptor::relu()
        }
    }

    fn create_convolution_bias_activation(&mut self, handle: &cudnn::Handle) -> Result<cudnn::ConvolutionBiasActivation, cudnn::Status> {
        let num_inputs = self.filter_shape[1];
        let num_outputs = self.filter_shape[0];

        cudnn::ConvolutionBiasActivation::new(
            handle,
            self.alpha[0],
            create_tensor_descriptor(self.batch_size, num_inputs, self.width_height)?,
            self.create_filter_descriptor()?,
            self.create_convolution_descriptor()?,
            self.alpha[1],
            create_offset_descriptor(num_outputs)?,
            self.create_activation_descriptor()?,
            create_tensor_descriptor(self.batch_size, num_outputs, self.width_height)?
        )
    }

    pub fn build(mut self, handle: &cudnn::Handle) -> Result<Conv2d, cudnn::Status> {
        Ok(Conv2d {
            conv_desc: self.create_convolution_bias_activation(handle)?,
            filter: self.filter.take().expect("no filter given"),
            offset: self.offset.take().expect("no offset given"),
        })
    }
}

impl Conv2d {
    pub fn new(batch_size: i32, filter: [i32; 4]) -> Conv2dBuilder {
        Conv2dBuilder::new(batch_size, filter)
    }

    pub fn offset(&self) -> &Tensor {
        &self.offset
    }

    pub fn prepare(&self, handle: &cudnn::Handle, stream: &cuda::Stream) -> Result<bool, Error> {
        handle.set_stream(stream)?;
        if self.filter.copy_to_device(&stream)? && self.offset.copy_to_device(&stream)? {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn forward<A: cuda::Allocator + Clone>(
        &self,
        handle: &cudnn::Handle,
        input: &cuda::SmartPtr<A>,
        allocator: &A,
        stream: &cuda::Stream
    ) -> Result<cuda::SmartPtr<A>, Error>
    {
        let workspace = cuda::malloc(self.conv_desc.fwd_algo_perf().memory(), allocator)?;
        let output = cuda::malloc(self.conv_desc.output().size_in_bytes()?, allocator)?;

        self.prepare(handle, stream)?;
        self.conv_desc.forward(
            handle,
            input.as_ptr(),
            self.filter.get().as_ptr(),
            workspace.as_ptr(), workspace.size_in_bytes(),
            output.as_ptr(),
            self.offset.get().as_ptr(),
            output.as_ptr()
        )?;

        Ok(output)
    }

    pub fn forward_skip<A: cuda::Allocator + Clone>(
        &self,
        handle: &cudnn::Handle,
        input: &cuda::SmartPtr<A>,
        skip_input: &cuda::SmartPtr<A>,
        allocator: &A,
        stream: &cuda::Stream
    ) -> Result<cuda::SmartPtr<A>, Error>
    {
        let workspace = cuda::malloc(self.conv_desc.fwd_algo_perf().memory(), allocator)?;
        let output = cuda::malloc(self.conv_desc.output().size_in_bytes()?, allocator)?;

        self.prepare(handle, stream)?;
        self.conv_desc.forward(
            handle,
            input.as_ptr(),
            self.filter.get().as_ptr(),
            workspace.as_ptr(), workspace.size_in_bytes(),
            skip_input.as_ptr(),
            self.offset.get().as_ptr(),
            output.as_ptr()
        )?;

        Ok(output)
    }
}

/// Returns true if the current device supports `f16` (in a
/// sensible way).
fn has_true_half() -> bool {
    let (version_major, version_minor) = cuda::Device::default().compute_capability().unwrap();

    (version_major == 6 && version_minor == 0) ||
        (version_major == 6 && version_minor == 2) ||
        (version_major >= 7)
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use dg_utils::types::f16;
    use super::*;

    fn create_tensor(data: &[f32]) -> Result<Tensor, Error> {
        Tensor::from_vec(data.iter().map(|&x| f16::from(x)).collect())
    }

    fn create_ptr<A: cuda::Allocator + Clone>(data: &[f32], allocator: &A) -> Result<cuda::SmartPtr<A>, Error> {
        let stream = cuda::Stream::default();
        let mut output = cuda::malloc(size_of::<f16>() * data.len(), allocator)?;
        output.copy_from_slice(&data.iter().map(|&x| f16::from(x)).collect::<Vec<_>>(), &stream)?;
        
        Ok(output)
    }

    #[test]
    fn pick_middle_and_sum() {
        let handle = cudnn::Handle::new().unwrap();
        let filter = create_tensor(&[
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]).unwrap();
        let offset = create_tensor(&[
            0.0, 0.0
        ]).unwrap();
        let conv2d = Conv2d::new(1, [2, 1, 3, 3])
            .with_filter(filter)
            .with_offset(offset)
            .with_width_height(3)
            .build(&handle)
            .unwrap();

        //
        let stream = cuda::Stream::default();
        let allocator = cuda::Native::default();
        let input = create_ptr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &allocator).unwrap();
        let output = conv2d.forward(&handle, &input, &allocator, &stream).unwrap();

        assert_eq!(
            vec! [
                 1.0, 12.0,
                 2.0, 21.0,
                 3.0, 16.0,
                 4.0, 27.0,
                 5.0, 45.0,
                 6.0, 33.0,
                 7.0, 24.0,
                 8.0, 39.0,
                 9.0, 28.0
            ],
            output.to_vec::<f16>(&stream)
                .unwrap()
                .into_iter().map(|x| f32::from(x)).collect::<Vec<_>>()
        )
    }
}
