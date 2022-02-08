// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use crate::{Allocator, AsSlice, Err, Variable, Io};
use super::{LayerFactory, LayerImpl};

use dg_cuda::{self as cuda, cudnn};

use std::collections::HashMap;

#[derive(Default)]
pub struct PredictionFactory;

impl LayerFactory for PredictionFactory {
    fn build(
        &self,
        _handle: &cudnn::Handle,
        _variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Result<Box<dyn LayerImpl>, Err>
    {
        Ok(Box::new(Prediction::new()?))
    }
}

pub struct Prediction {
    kernel: cuda::PerDevice<[cuda::Ptr; 2]>,
    offset: cuda::PerDevice<[cuda::Ptr; 2]>,
    conv_desc: cuda::PerDevice<HashMap<i32, [cudnn::ConvolutionBiasActivation; 2]>>,
    softmax: cuda::PerDevice<HashMap<i32, cudnn::Softmax>>
}

impl Prediction {
    pub fn new() -> Result<Self, Err> {
        Ok(Self {
            kernel: cuda::PerDevice::new()?,
            offset: cuda::PerDevice::new()?,
            conv_desc: cuda::PerDevice::new()?,
            softmax: cuda::PerDevice::new()?,
        })
    }

    fn create_softmax(batch_size: i32, shape: &[i32]) -> Result<cudnn::Softmax, cudnn::Status> {
        cudnn::Softmax::new(
            cudnn::SoftmaxMode::Instance,
            Self::create_output_descriptor(batch_size, shape)?,
            Self::create_output_descriptor(batch_size, shape)?,
            [1.0, 0.0]
        )
    }

    fn create_dense_bias_activation(handle: &cudnn::Handle, batch_size: i32, shape: &[i32], act_desc: cudnn::ActivationDescriptor) -> Result<cudnn::ConvolutionBiasActivation, cudnn::Status> {
        assert_eq!(shape.len(), 2, "filter shape must be 2 elements, received {:?}", shape);

        cudnn::ConvolutionBiasActivation::new(
            handle,
            1.0,
            Self::create_input_descriptor(batch_size, shape)?,
            Self::create_filter_descriptor(shape)?,
            Self::create_dense_descriptor()?,
            0.0,
            Self::create_offset_descriptor(shape)?,
            act_desc,
            Self::create_output_descriptor(batch_size, shape)?,
            Self::create_output_descriptor(batch_size, shape)?
        )
    }

    fn create_dense_descriptor() -> Result<cudnn::ConvolutionDescriptor, cudnn::Status> {
        let dense_desc = cudnn::ConvolutionDescriptor::new(
            [0, 0],
            [1, 1],
            [1, 1],
            cudnn::ConvolutionMode::CrossCorrelation,
            cudnn::DataType::Half
        )?;

        Ok(dense_desc)
    }

    fn create_filter_descriptor(shape: &[i32]) -> Result<cudnn::FilterDescriptor, cudnn::Status> {
        debug_assert_eq!(shape.len(), 2, "filter shape must be 2 elements, received {:?}", shape);

        cudnn::FilterDescriptor::new(
            cudnn::DataType::Half,
            cudnn::TensorFormat::NHWC,
            [shape[0], shape[1], 1, 1]
        )
    }

    fn create_offset_descriptor(shape: &[i32]) -> Result<cudnn::TensorDescriptor, cudnn::Status> {
        debug_assert_eq!(shape.len(), 2, "kernel shape must be 2 elements, received {:?}", shape);

        cudnn::TensorDescriptor::new(
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Half,
            [1, shape[0], 1, 1]
        )
    }

    fn create_input_descriptor(batch_size: i32, shape: &[i32]) -> Result<cudnn::TensorDescriptor, cudnn::Status> {
        debug_assert_eq!(shape.len(), 2, "kernel shape must be 2 elements, received {:?}", shape);

        cudnn::TensorDescriptor::new(
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Half,
            [batch_size, shape[1], 1, 1]
        )
    }

    fn create_output_descriptor(batch_size: i32, shape: &[i32]) -> Result<cudnn::TensorDescriptor, cudnn::Status> {
        debug_assert_eq!(shape.len(), 2, "kernel shape must be 2 elements, received {:?}", shape);

        cudnn::TensorDescriptor::new(
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Half,
            [batch_size, shape[0], 1, 1]
        )
    }
}

impl LayerImpl for Prediction {
    fn build(
        &mut self,
        _handle: &cudnn::Handle,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        *self.kernel = [
            variables.get("policy/linear_1").ok_or_else(|| Err::MissingVariable("policy/linear_1".to_string()))?.as_ptr(stream)?,
            variables.get("value/linear_1").ok_or_else(|| Err::MissingVariable("value/linear_1".to_string()))?.as_ptr(stream)?
        ];
        *self.offset = [
            variables.get("policy/linear_1/offset").ok_or_else(|| Err::MissingVariable("policy/linear_1/offset".to_string()))?.as_ptr(stream)?,
            variables.get("value/linear_1/offset").ok_or_else(|| Err::MissingVariable("value/linear_1/offset".to_string()))?.as_ptr(stream)?
        ];

        Ok(())
    }

    fn prepare(
        &mut self,
        handle: &cudnn::Handle,
        batch_size: i32,
        variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        let shape_1: &[i32] = variables.get("policy/linear_1/shape").ok_or_else(|| Err::MissingVariable("policy/linear_1/shape".to_string()))?.as_slice()?;
        let shape_2: &[i32] = variables.get("value/linear_1/shape").ok_or_else(|| Err::MissingVariable("value/linear_1/shape".to_string()))?.as_slice()?;

        if !self.conv_desc.contains_key(&batch_size) {
            self.conv_desc.insert(batch_size, [
                Self::create_dense_bias_activation(handle, batch_size, shape_1, cudnn::ActivationDescriptor::identity()?)?,
                Self::create_dense_bias_activation(handle, batch_size, shape_2, cudnn::ActivationDescriptor::tanh()?)?,
            ]);
            self.softmax.insert(batch_size, Self::create_softmax(batch_size, shape_1)?);
        }

        Ok(())
    }

    fn forward(
        &self,
        handle: &cudnn::Handle,
        inputs: Io,
        allocator: &mut Allocator,
        _stream: &cuda::Stream,
    ) -> Result<Io, Err>
    {
        let conv_desc = &self.conv_desc[&(inputs.batch_size as i32)];
        let softmax = &self.softmax[&(inputs.batch_size as i32)];
        let linear_policy = cuda::malloc(conv_desc[0].output().size_in_bytes()?, allocator)?;
        let workspace_size_in_bytes = conv_desc[0].fwd_algo_perf().memory().max(conv_desc[1].fwd_algo_perf().memory());
        let workspace = cuda::malloc(workspace_size_in_bytes, allocator)?;

        conv_desc[0].forward(
            handle,
            inputs.current().as_ptr(),
            self.kernel[0].as_ptr(),
            workspace.as_ptr(),
            workspace.size_in_bytes(),
            linear_policy.as_ptr(),
            self.offset[0].as_ptr(),
            linear_policy.as_ptr()
        )?;

        softmax.forward(
            handle,
            linear_policy.as_ptr(),
            inputs.policy.as_ptr()
        )?;

        conv_desc[1].forward(
            handle,
            inputs.current().as_ptr(),
            self.kernel[1].as_ptr(),
            workspace.as_ptr(),
            workspace.size_in_bytes(),
            inputs.value.as_ptr(),
            self.offset[1].as_ptr(),
            inputs.value.as_ptr()
        )?;

        Ok(inputs)
    }
}

#[cfg(test)]
mod tests {
    // pass
}
