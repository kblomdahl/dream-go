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
pub struct ResidualBlockFactory;

impl LayerFactory for ResidualBlockFactory {
    fn build(
        &self,
        _handle: &cudnn::Handle,
        _variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Result<Box<dyn LayerImpl>, Err>
    {
        Ok(Box::new(ResidualBlock::new()?))
    }
}

pub struct ResidualBlock {
    filter: cuda::PerDevice<[cuda::Ptr; 2]>,
    offset: cuda::PerDevice<[cuda::Ptr; 2]>,

    conv_desc: cuda::PerDevice<HashMap<i32, [cudnn::ConvolutionBiasActivation; 2]>>
}

impl ResidualBlock {
    pub fn new() -> Result<Self, Err> {
        Ok(Self {
            filter: cuda::PerDevice::<_>::new()?,
            offset: cuda::PerDevice::<_>::new()?,
            conv_desc: cuda::PerDevice::<_>::new()?,
        })
    }

    fn create_convolution_bias_activation(handle: &cudnn::Handle, batch_size: i32, alpha: [f32; 2], shape: &[i32]) -> Result<cudnn::ConvolutionBiasActivation, cudnn::Status> {
        assert_eq!(shape.len(), 4, "filter shape must be 4 elements, received {:?}", shape);

        cudnn::ConvolutionBiasActivation::new(
            handle,
            alpha[0],
            Self::create_input_descriptor(batch_size, shape)?,
            Self::create_filter_descriptor(shape)?,
            Self::create_convolution_descriptor(shape)?,
            alpha[1],
            Self::create_offset_descriptor(shape)?,
            cudnn::ActivationDescriptor::relu()?,
            Self::create_output_descriptor(batch_size, shape)?,
            Self::create_output_descriptor(batch_size, shape)?
        )
    }

    fn create_convolution_descriptor(shape: &[i32]) -> Result<cudnn::ConvolutionDescriptor, cudnn::Status> {
        debug_assert_eq!(shape.len(), 4, "filter shape must be 4 elements, received {:?}", shape);

        cudnn::ConvolutionDescriptor::new(
            [shape[1] / 2, shape[2] / 2],
            [1, 1],
            [1, 1],
            cudnn::ConvolutionMode::CrossCorrelation,
            cudnn::DataType::Half
        )
    }

    fn create_filter_descriptor(shape: &[i32]) -> Result<cudnn::FilterDescriptor, cudnn::Status> {
        debug_assert_eq!(shape.len(), 4, "filter shape must be 4 elements, received {:?}", shape);

        cudnn::FilterDescriptor::new(
            cudnn::DataType::Half,
            cudnn::TensorFormat::NHWC,
            [shape[0], shape[3], shape[1], shape[2]]
        )
    }

    fn create_offset_descriptor(shape: &[i32]) -> Result<cudnn::TensorDescriptor, cudnn::Status> {
        debug_assert_eq!(shape.len(), 4, "filter shape must be 4 elements, received {:?}", shape);

        cudnn::TensorDescriptor::new(
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Half,
            [1, shape[0], 1, 1]
        )
    }

    fn create_input_descriptor(batch_size: i32, shape: &[i32]) -> Result<cudnn::TensorDescriptor, cudnn::Status> {
        debug_assert_eq!(shape.len(), 4, "filter shape must be 4 elements, received {:?}", shape);

        cudnn::TensorDescriptor::new(
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Half,
            [batch_size, shape[3], 19, 19]
        )
    }

    fn create_output_descriptor(batch_size: i32, shape: &[i32]) -> Result<cudnn::TensorDescriptor, cudnn::Status> {
        debug_assert_eq!(shape.len(), 4, "filter shape must be 4 elements, received {:?}", shape);

        cudnn::TensorDescriptor::new(
            cudnn::TensorFormat::NHWC,
            cudnn::DataType::Half,
            [batch_size, shape[0], 19, 19]
        )
    }
}

impl LayerImpl for ResidualBlock {
    fn build(
        &mut self,
        _handle: &cudnn::Handle,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        *self.filter = [
            variables.get("conv_1").ok_or_else(|| Err::MissingVariable("conv_1".to_string()))?.as_ptr(stream)?,
            variables.get("conv_2").ok_or_else(|| Err::MissingVariable("conv_2".to_string()))?.as_ptr(stream)?
        ];
        *self.offset = [
            variables.get("conv_1/offset").ok_or_else(|| Err::MissingVariable("conv_1/offset".to_string()))?.as_ptr(stream)?,
            variables.get("conv_2/offset").ok_or_else(|| Err::MissingVariable("conv_2/offset".to_string()))?.as_ptr(stream)?
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
        if !self.conv_desc.contains_key(&batch_size) {
            let shape_1: &[i32] = variables.get("conv_1/shape").ok_or_else(|| Err::MissingVariable("conv_1/shape".to_string()))?.as_slice()?;
            let shape_2: &[i32] = variables.get("conv_2/shape").ok_or_else(|| Err::MissingVariable("conv_2/shape".to_string()))?.as_slice()?;

            self.conv_desc.insert(batch_size, [
                Self::create_convolution_bias_activation(handle, batch_size, [1.0, 0.0], shape_1)?,
                Self::create_convolution_bias_activation(handle, batch_size, [1.0, 1.0], shape_2)?
            ]);
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
        let workspace_size_in_bytes = conv_desc[0].fwd_algo_perf().memory().max(conv_desc[1].fwd_algo_perf().memory());
        let workspace = cuda::malloc(workspace_size_in_bytes, allocator)?;
        let output_0 = cuda::malloc(conv_desc[0].output().size_in_bytes()?, allocator)?;
        let intermediate = inputs.current().as_ptr();

        conv_desc[0].forward(
            handle,
            intermediate,
            self.filter[0].as_ptr(),
            workspace.as_ptr(),
            workspace.size_in_bytes(),
            output_0.as_ptr(),
            self.offset[0].as_ptr(),
            output_0.as_ptr()
        )?;

        conv_desc[1].forward(
            handle,
            output_0.as_ptr(),
            self.filter[1].as_ptr(),
            workspace.as_ptr(),
            workspace.size_in_bytes(),
            intermediate,
            self.offset[1].as_ptr(),
            intermediate
        )?;

        Ok(inputs)
    }
}

#[cfg(test)]
mod tests {
    // pass
}
