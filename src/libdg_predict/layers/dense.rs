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
pub struct DenseFactory;

impl LayerFactory for DenseFactory {
    fn build(
        &self,
        _handle: &cudnn::Handle,
        _variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Result<Box<dyn LayerImpl>, Err>
    {
        Ok(Box::new(Dense::new()?))
    }
}

pub struct Dense {
    kernel: cuda::PerDevice<cuda::Ptr>,
    offset: cuda::PerDevice<cuda::Ptr>,

    conv_desc: HashMap<i32, cudnn::ConvolutionBiasActivation>
}

impl Dense {
    pub fn new() -> Result<Self, Err> {
        Ok(Self {
            kernel: cuda::PerDevice::<_>::new()?,
            offset: cuda::PerDevice::<_>::new()?,
            conv_desc: HashMap::new()
        })
    }

    fn create_dense_bias_activation(handle: &cudnn::Handle, batch_size: i32, shape: &[i32]) -> Result<cudnn::ConvolutionBiasActivation, cudnn::Status> {
        assert_eq!(shape.len(), 2, "filter shape must be 2 elements, received {:?}", shape);

        cudnn::ConvolutionBiasActivation::new(
            handle,
            1.0,
            Self::create_input_descriptor(batch_size, shape)?,
            Self::create_filter_descriptor(shape)?,
            Self::create_dense_descriptor()?,
            0.0,
            Self::create_offset_descriptor(shape)?,
            cudnn::ActivationDescriptor::relu()?,
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
            cudnn::DataType::Float
        )?;

        // when using tensor cores the output is all zeros, so do not use them for now
        dense_desc.set_default_math_type()?;
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

impl LayerImpl for Dense {
    fn build(
        &mut self,
        _handle: &cudnn::Handle,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        *self.kernel = variables.get("kernel").ok_or_else(|| Err::MissingVariable("kernel".to_string()))?.as_ptr(stream)?;
        *self.offset = variables.get("offset").ok_or_else(|| Err::MissingVariable("offset".to_string()))?.as_ptr(stream)?;

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
        let shape: &[i32] = variables.get("shape").ok_or_else(|| Err::MissingVariable("shape".to_string()))?.as_slice()?;

        if !self.conv_desc.contains_key(&batch_size) {
            self.conv_desc.insert(batch_size, Self::create_dense_bias_activation(handle, batch_size, shape)?);
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
        let fwd_algo_perf = conv_desc.fwd_algo_perf();
        let workspace = cuda::malloc(fwd_algo_perf.memory(), allocator)?;
        let output = cuda::malloc(conv_desc.output().size_in_bytes()?, allocator)?;

        conv_desc.forward(
            handle,
            inputs.current().as_ptr(),
            self.kernel.as_ptr(),
            workspace.as_ptr(),
            workspace.size_in_bytes(),
            output.as_ptr(),
            self.offset.as_ptr(),
            output.as_ptr()
        )?;

        Ok(inputs.with_intermediate(output))
    }
}


#[cfg(test)]
mod tests {
    use crate::{Config, ExecutionPlan};
    use super::*;

    use dg_utils::types::f16;

    const INPUT: [f32; 4] = [
        1.0, 2.0, 3.0, 4.0
    ];

    const OUTPUT: [f32; 2] = [
        31.0,
        12.0
    ];

    const FILTER: [f32; 8] = [
        1.0, 2.0, 3.0, 4.0,
        1.0, 1.0, 1.0, 1.0,
    ];

    const OFFSET: [f32; 2] = [
        1.0,
        2.0
    ];

    #[test]
    fn factory() -> Result<(), Err> {
        let factory = DenseFactory::default();
        let config = Config::default().with_num_features(4).with_image_size(1);
        let mut plan = ExecutionPlan::new(&cuda::Concurrent::<cuda::Sticky<cuda::Native>>::default())?;
        let features = INPUT.iter().map(|x| f16::from(*x)).collect::<Vec<_>>();
        let filter = FILTER.iter().map(|x| f16::from(*x)).collect::<Vec<_>>();
        let offset = OFFSET.iter().map(|x| f16::from(*x)).collect::<Vec<_>>();
        let mut io = Io::new(1, &config, &mut plan.allocator)?
            .copy_features_from(&features, &plan)?;
        let variables = HashMap::from([
            ("shape".to_string(), Variable::from(vec! [2, 4])),
            ("kernel".to_string(), Variable::from(filter)),
            ("offset".to_string(), Variable::from(offset)),
        ]);
        let mut layer = factory.build(&plan.handle, &variables, &plan.stream)?;
        layer.build(&plan.handle, &variables, &plan.stream)?;
        layer.prepare(&plan.handle, 1, &variables, &plan.stream)?;
        io = layer.forward(&plan.handle, io, &mut plan.allocator, &plan.stream)?;

        assert_eq!(
            io.current().to_vec::<f16>(&plan.stream)?.into_iter().map(|x| f32::from(x)).collect::<Vec<_>>(),
            OUTPUT
        );

        Ok(())
    }
}
