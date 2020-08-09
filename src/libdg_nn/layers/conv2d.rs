
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

use crate::ffi::*;
use crate::tensor::Tensor;
use crate::layers::{create_tensor_descriptor, create_offset_descriptor};
use crate::Error;

pub struct Conv2d {
    conv_desc: cudnn2::ConvolutionBiasActivation,
    filter: Tensor,
    offset: Tensor
}

pub struct Conv2dBuilder {
    batch_size: i32,
    filter_shape: [i32; 4],
    alpha: [f32; 2],
    act_desc: Option<cudnn2::ActivationDescriptor>,
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
            filter_shape: filter,
            alpha: [1.0, 0.0],
            act_desc: None,
            filter: None,
            offset: None
        }
    }

    pub fn with_tensors(mut self, tensors: &HashMap<String, Tensor>, name: &str) -> Self {
        self.filter = Some(tensors.get(&format!("{}:0", name)).cloned().expect("no filter available"));
        self.offset = Some(tensors.get(&format!("{}/offset:0", name)).cloned().expect("no offset available"));
        self
    }

    pub fn with_alpha(mut self, alpha: [f32; 2]) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_activation(mut self, act_desc: cudnn2::ActivationDescriptor) -> Self {
        self.act_desc = Some(act_desc);
        self
    }

    fn create_convolution_descriptor(&self) -> Result<cudnn2::ConvolutionDescriptor, cudnn2::Status> {
        let filter_size = self.filter_shape[2];

        cudnn2::ConvolutionDescriptor::new(
            &[filter_size / 2, filter_size / 2], // padding
            &[1, 1], // stride
            &[1, 1], // dilation
            cudnn2::ConvolutionMode::CrossCorrelation,
            if has_true_half() { cudnn2::DataType::Half } else { cudnn2::DataType::Float }
        )
    }

    fn create_filter_descriptor(&self) -> Result<cudnn2::FilterDescriptor, cudnn2::Status> {
        cudnn2::FilterDescriptor::new(
            cudnn2::DataType::Half,
            cudnn2::TensorFormat::NHWC,
            &self.filter_shape
        )
    }

    fn create_activation_descriptor(&mut self) -> Result<cudnn2::ActivationDescriptor, cudnn2::Status> {
        if let Some(act) = self.act_desc.take() {
            Ok(act)
        } else {
            cudnn2::ActivationDescriptor::relu()
        }
    }

    fn create_convolution_bias_activation(&mut self, handle: &cudnn2::Handle) -> Result<cudnn2::ConvolutionBiasActivation, cudnn2::Status> {
        let num_inputs = self.filter_shape[1];
        let num_outputs = self.filter_shape[0];

        cudnn2::ConvolutionBiasActivation::new(
            handle,
            self.alpha[0],
            create_tensor_descriptor(self.batch_size, num_inputs)?,
            self.create_filter_descriptor()?,
            self.create_convolution_descriptor()?,
            self.alpha[1],
            create_offset_descriptor(num_outputs)?,
            self.create_activation_descriptor()?,
            create_tensor_descriptor(self.batch_size, num_outputs)?
        )
    }

    pub fn build(mut self, handle: &cudnn2::Handle) -> Result<Conv2d, cudnn2::Status> {
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

    pub fn prepare(&self, handle: &cudnn2::Handle, stream: &cuda2::Stream) -> Result<bool, Error> {
        handle.set_stream(stream)?;
        if self.filter.copy_to_device(&stream)? && self.offset.copy_to_device(&stream)? {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn forward<A: cuda2::Allocator + Clone>(
        &self,
        handle: &cudnn2::Handle,
        input: &cuda2::SmartPtr<A>,
        allocator: &A,
        stream: &cuda2::Stream
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        let workspace = cuda2::malloc(self.conv_desc.fwd_algo_perf().memory(), allocator)?;
        let output = cuda2::malloc(self.conv_desc.output().size_in_bytes()?, allocator)?;

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

    pub fn forward_skip<A: cuda2::Allocator + Clone>(
        &self,
        handle: &cudnn2::Handle,
        input: &cuda2::SmartPtr<A>,
        skip_input: &cuda2::SmartPtr<A>,
        allocator: &A,
        stream: &cuda2::Stream
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        let workspace = cuda2::malloc(self.conv_desc.fwd_algo_perf().memory(), allocator)?;
        let output = cuda2::malloc(self.conv_desc.output().size_in_bytes()?, allocator)?;

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
    let mut version_major: i32 = 0;
    let mut version_minor: i32 = 0;

    unsafe {
        assert!(cuda::cudaDeviceGetAttribute(&mut version_major, cuda::DeviceAttr::ComputeCapabilityMajor, 0).is_ok());
        assert!(cuda::cudaDeviceGetAttribute(&mut version_minor, cuda::DeviceAttr::ComputeCapabilityMinor, 0).is_ok());
    }

    (version_major == 6 && version_minor == 0) ||
        (version_major == 6 && version_minor == 2) ||
        (version_major >= 7)
}

#[cfg(test)]
mod tests {
    // pass
}
