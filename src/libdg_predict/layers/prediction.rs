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
pub struct PredictionFactory;

impl LayerFactory for PredictionFactory {
    fn build(
        &self,
        _handle: &cudnn::Handle,
        _variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Box<dyn LayerImpl>
    {
        Box::new(Prediction::new())
    }
}

pub struct Prediction {
    kernel: [cuda::Ptr; 2],
    offset: [cuda::Ptr; 2],
    conv_desc: HashMap<i32, [cublas_lt::Matmul<f16>; 2]>,
    softmax: HashMap<i32, cudnn::Softmax>,
    tanh: HashMap<i32, cudnn::Activation>
}

impl Prediction {
    pub fn new() -> Self {
        Self {
            kernel: [cuda::Ptr::default(), cuda::Ptr::default()],
            offset: [cuda::Ptr::default(), cuda::Ptr::default()],
            conv_desc: HashMap::new(),
            softmax: HashMap::new(),
            tanh: HashMap::new(),
        }
    }

    fn create_softmax(batch_size: i32, shape: &[i32]) -> Result<cudnn::Softmax, cudnn::Status> {
        cudnn::Softmax::new(
            cudnn::SoftmaxMode::Instance,
            Self::create_output_tensor_descriptor(batch_size, shape)?,
            Self::create_output_tensor_descriptor(batch_size, shape)?,
            [1.0, 0.0]
        )
    }

    fn create_tanh(batch_size: i32, shape: &[i32]) -> Result<cudnn::Activation, cudnn::Status> {
        cudnn::Activation::new(
            cudnn::ActivationDescriptor::tanh()?,
            Self::create_output_tensor_descriptor(batch_size, shape)?,
            Self::create_output_tensor_descriptor(batch_size, shape)?,
            [1.0, 0.0]
        )
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
            m.with_epilogue(cublas_lt::Epilogue::Bias)?
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

    fn create_output_tensor_descriptor(batch_size: i32, shape: &[i32]) -> Result<cudnn::TensorDescriptor, cudnn::Status> {
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
        _light_handle: &cublas_lt::Handle,
        _handle: &cudnn::Handle,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        self.kernel = [
            variables.get("policy/linear_1").ok_or_else(|| Err::MissingVariable("policy/linear_1".to_string()))?.as_ptr(stream)?,
            variables.get("value/linear_1").ok_or_else(|| Err::MissingVariable("value/linear_1".to_string()))?.as_ptr(stream)?
        ];
        self.offset = [
            variables.get("policy/linear_1/offset").ok_or_else(|| Err::MissingVariable("policy/linear_1/offset".to_string()))?.as_ptr(stream)?,
            variables.get("value/linear_1/offset").ok_or_else(|| Err::MissingVariable("value/linear_1/offset".to_string()))?.as_ptr(stream)?
        ];

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
        let shape_1: &[i32] = variables.get("policy/linear_1/shape").ok_or_else(|| Err::MissingVariable("policy/linear_1/shape".to_string()))?.as_slice()?;
        let shape_2: &[i32] = variables.get("value/linear_1/shape").ok_or_else(|| Err::MissingVariable("value/linear_1/shape".to_string()))?.as_slice()?;

        if !self.conv_desc.contains_key(&batch_size) {
            self.conv_desc.insert(batch_size, [
                Self::create_matmul(light_handle, batch_size, shape_1)?,
                Self::create_matmul(light_handle, batch_size, shape_2)?,
            ]);
            self.softmax.insert(batch_size, Self::create_softmax(batch_size, shape_1)?);
            self.tanh.insert(batch_size, Self::create_tanh(batch_size, shape_2)?);
        }

        Ok(())
    }

    fn forward(
        &self,
        light_handle: &cublas_lt::Handle,
        handle: &cudnn::Handle,
        inputs: Io,
        allocator: &mut Allocator,
        stream: &cuda::Stream,
    ) -> Result<Io, Err>
    {
        let conv_desc = &self.conv_desc[&(inputs.batch_size as i32)];
        let softmax = &self.softmax[&(inputs.batch_size as i32)];
        let tanh = &self.tanh[&(inputs.batch_size as i32)];
        let linear_policy = cuda::malloc(conv_desc[0].d().size_in_bytes()?, allocator)?;
        let linear_value = cuda::malloc(conv_desc[1].d().size_in_bytes()?, allocator)?;
        let workspace_size_in_bytes = conv_desc[0].algo().memory().max(conv_desc[1].algo().memory());
        let workspace = cuda::malloc(workspace_size_in_bytes, allocator)?;

        conv_desc[0].desc().set_bias(&self.offset[0])?;
        conv_desc[0].forward(
            light_handle,
            self.kernel[0].as_ptr(),
            inputs.current().as_ptr(),
            linear_policy.as_ptr(),
            linear_policy.as_ptr(),
            workspace.as_ptr(),
            workspace.size_in_bytes(),
            &stream
        )?;

        softmax.forward(
            handle,
            linear_policy.as_ptr(),
            inputs.policy.as_ptr()
        )?;

        conv_desc[1].desc().set_bias(&self.offset[1])?;
        conv_desc[1].forward(
            light_handle,
            self.kernel[1].as_ptr(),
            inputs.current().as_ptr(),
            linear_value.as_ptr(),
            linear_value.as_ptr(),
            workspace.as_ptr(),
            workspace.size_in_bytes(),
            &stream
        )?;

        tanh.forward(
            handle,
            linear_value.as_ptr(),
            inputs.value.as_ptr()
        )?;

        Ok(inputs)
    }
}

#[cfg(test)]
mod tests {
    // pass
}
