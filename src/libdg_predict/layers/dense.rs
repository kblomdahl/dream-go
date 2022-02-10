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

use dg_utils::types::f16;
use dg_cuda::{self as cuda, cudnn, cublas_lt};

use std::collections::HashMap;

#[derive(Default)]
pub struct DenseFactory;

impl LayerFactory for DenseFactory {
    fn build(
        &self,
        _handle: &cudnn::Handle,
        _variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Box<dyn LayerImpl>
    {
        Box::new(Dense::new())
    }
}

pub struct Dense {
    kernel: cuda::Ptr,
    offset: cuda::Ptr,

    matmul_desc: HashMap<i32, cublas_lt::Matmul::<f16>>
}

impl Dense {
    pub fn new() -> Self {
        Self {
            kernel: cuda::Ptr::default(),
            offset: cuda::Ptr::default(),
            matmul_desc: HashMap::new()
        }
    }

    fn create_matmul(handle: &cublas_lt::Handle, batch_size: i32, shape: &[i32]) -> Result<cublas_lt::Matmul<f16>, cublas_lt::Status> {
        assert_eq!(shape.len(), 2, "filter shape must be 2 elements, received {:?}", shape);

        cublas_lt::Matmul::<f16>::new(
            handle,
            Self::create_matmul_descriptor()?,
            [1.0, 0.0],
            Self::create_kernel_descriptor(shape)?,
            Self::create_input_descriptor(batch_size, shape)?,
            Self::create_output_descriptor(batch_size, shape)?,
            Self::create_output_descriptor(batch_size, shape)?,
        )
    }

    fn create_matmul_descriptor() -> Result<cublas_lt::MatmulDesc, cublas_lt::Status> {
        cublas_lt::MatmulDesc::new(
            cublas_lt::ComputeType::Real16F,
            cublas_lt::DataType::Real16F
        ).and_then(|m| {
            m.with_epilogue(cublas_lt::Epilogue::ReluBias)?
             .with_transpose_a(cublas_lt::Operation::Transpose)?
             .with_transpose_b(cublas_lt::Operation::NonTranspose)
        })
    }

    fn create_kernel_descriptor(shape: &[i32]) -> Result<cublas_lt::MatrixLayout, cublas_lt::Status> {
        debug_assert_eq!(shape.len(), 2, "kernel shape must be 2 elements, received {:?}", shape);

        cublas_lt::MatrixLayout::new(
            cublas_lt::DataType::Real16F,
            shape[1] as u64,
            shape[0] as u64,
            shape[1] as i64
        )
    }

    fn create_input_descriptor(batch_size: i32, shape: &[i32]) -> Result<cublas_lt::MatrixLayout, cublas_lt::Status> {
        debug_assert_eq!(shape.len(), 2, "kernel shape must be 2 elements, received {:?}", shape);

        cublas_lt::MatrixLayout::new(
            cublas_lt::DataType::Real16F,
            shape[1] as u64,
            batch_size as u64,
            shape[1] as i64
        )
    }

    fn create_output_descriptor(batch_size: i32, shape: &[i32]) -> Result<cublas_lt::MatrixLayout, cublas_lt::Status> {
        debug_assert_eq!(shape.len(), 2, "kernel shape must be 2 elements, received {:?}", shape);

        cublas_lt::MatrixLayout::new(
            cublas_lt::DataType::Real16F,
            shape[0] as u64,
            batch_size as u64,
            shape[0] as i64
        )
    }
}

impl LayerImpl for Dense {
    fn build(
        &mut self,
        _light_handle: &cublas_lt::Handle,
        _handle: &cudnn::Handle,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        self.kernel = variables.get("kernel").ok_or_else(|| Err::MissingVariable("kernel".to_string()))?.as_ptr(stream)?;
        self.offset = variables.get("offset").ok_or_else(|| Err::MissingVariable("offset".to_string()))?.as_ptr(stream)?;

        Ok(())
    }

    fn prepare(
        &mut self,
        light_handle: &cublas_lt::Handle,
        _handle: &cudnn::Handle,
        batch_size: i32,
        variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        if !self.matmul_desc.contains_key(&batch_size) {
            let shape: &[i32] = variables.get("shape").ok_or_else(|| Err::MissingVariable("shape".to_string()))?.as_slice()?;

            self.matmul_desc.insert(batch_size, Self::create_matmul(light_handle, batch_size, shape)?);
        }

        Ok(())
    }

    fn forward(
        &self,
        light_handle: &cublas_lt::Handle,
        _handle: &cudnn::Handle,
        inputs: Io,
        allocator: &mut Allocator,
        stream: &cuda::Stream,
    ) -> Result<Io, Err>
    {
        let matmul_desc = &self.matmul_desc[&(inputs.batch_size as i32)];
        let workspace = cuda::malloc(matmul_desc.algo().memory(), allocator)?;
        let output = cuda::malloc(matmul_desc.d().size_in_bytes()?, allocator)?;

        matmul_desc.desc().set_bias(&self.offset)?;
        matmul_desc.forward(
            light_handle,
            self.kernel.as_ptr(),
            inputs.current().as_ptr(),
            output.as_ptr(),
            output.as_ptr(),
            workspace.as_ptr(),
            workspace.size_in_bytes(),
            stream
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
