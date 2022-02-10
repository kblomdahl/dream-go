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

use crate::{Allocator, Err, Variable, AsSlice, Io};
use super::{LayerFactory, LayerImpl};

use dg_cuda::{self as cuda, cudnn, cublas_lt};

use std::collections::HashMap;

#[derive(Default)]
pub struct ConvFactory;

impl LayerFactory for ConvFactory {
    fn build(
        &self,
        handle: &cudnn::Handle,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Box<dyn LayerImpl>
    {
        Box::new(Conv::new(handle, variables, stream))
    }
}

pub struct Conv {
    filter: cuda::Ptr,
    offset: cuda::Ptr,

    conv_desc: HashMap<i32, cudnn::ConvolutionBiasActivation>
}

impl Conv {
    pub fn new(
        _handle: &cudnn::Handle,
        _variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Self
    {
        Self {
            filter: cuda::Ptr::default(),
            offset: cuda::Ptr::default(),
            conv_desc: HashMap::new(),
        }
    }

    fn create_convolution_bias_activation(handle: &cudnn::Handle, batch_size: i32, shape: &[i32]) -> Result<cudnn::ConvolutionBiasActivation, cudnn::Status> {
        assert_eq!(shape.len(), 4, "filter shape must be 4 elements, received {:?}", shape);

        cudnn::ConvolutionBiasActivation::new(
            handle,
            1.0,
            Self::create_input_descriptor(batch_size, shape)?,
            Self::create_filter_descriptor(shape)?,
            Self::create_convolution_descriptor(shape)?,
            0.0,
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

impl LayerImpl for Conv {
    fn build(
        &mut self,
        _light_handle: &cublas_lt::Handle,
        _handle: &cudnn::Handle,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        self.filter = variables.get("filter").ok_or_else(|| Err::MissingVariable("filter".to_string()))?.as_ptr(stream)?;
        self.offset = variables.get("offset").ok_or_else(|| Err::MissingVariable("offset".to_string()))?.as_ptr(stream)?;

        Ok(())
    }

    fn prepare(
        &mut self,
        _light_handle: &cublas_lt::Handle,
        handle: &cudnn::Handle,
        batch_size: i32,
        variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        let shape: &[i32] = variables.get("shape").ok_or_else(|| Err::MissingVariable("shape".to_string()))?.as_slice()?;

        if !self.conv_desc.contains_key(&batch_size) {
            self.conv_desc.insert(batch_size, Self::create_convolution_bias_activation(handle, batch_size, shape)?);
        }

        Ok(())
    }

    fn forward(
        &self,
        _light_handle: &cublas_lt::Handle,
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
            self.filter.as_ptr(),
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

    const INPUT: [f32; 9] = [
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ];

    const OUTPUT: [f32; 18] = [
        13.0, 26.0,
        22.0, 44.0,
        17.0, 34.0,
        28.0, 56.0,
        46.0, 92.0,
        34.0, 68.0,
        25.0, 50.0,
        40.0, 80.0,
        29.0, 58.0
    ];

    const FILTER: [f32; 18] = [
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        2.0, 2.0, 2.0,
        2.0, 2.0, 2.0,
        2.0, 2.0, 2.0
    ];

    const OFFSET: [f32; 2] = [
        1.0,
        2.0
    ];

    #[test]
    #[ignore] // TODO Allow the `image_size` to be set to 3 for this to work
    fn factory() -> Result<(), Err> {
        let factory = ConvFactory::default();
        let config = Config::default().with_num_features(1).with_image_size(3);
        let mut plan = ExecutionPlan::new(&cuda::Concurrent::<cuda::Sticky<cuda::Native>>::default())?;
        let features = INPUT.iter().map(|x| f16::from(*x)).collect::<Vec<_>>();
        let filter = FILTER.iter().map(|x| f16::from(*x)).collect::<Vec<_>>();
        let offset = OFFSET.iter().map(|x| f16::from(*x)).collect::<Vec<_>>();
        let mut io = Io::new(1, &config, &mut plan.allocator)?
            .copy_features_from(&features, &plan)?;
        let variables = HashMap::from([
            ("shape".to_string(), Variable::from(vec! [2, 3, 3, 1])),
            ("filter".to_string(), Variable::from(filter)),
            ("offset".to_string(), Variable::from(offset)),
        ]);
        let mut layer = factory.build(&plan.handle, &variables, &plan.stream);
        layer.build(&plan.light_handle, &plan.handle, &variables, &plan.stream)?;
        layer.prepare(&plan.light_handle, &plan.handle, 1, &variables, &plan.stream)?;
        io = layer.forward(&plan.light_handle, &plan.handle, io, &mut plan.allocator, &plan.stream)?;

        assert_eq!(
            io.current().to_vec::<f16>(&plan.stream)?.into_iter().map(|x| f32::from(x)).collect::<Vec<_>>(),
            OUTPUT
        );

        Ok(())
    }
}
