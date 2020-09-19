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
use crate::layers::{create_dense_descriptor, create_offset_descriptor};
use crate::Error;

pub struct Dense {
    transpose_filter: cudnn::Transform,
    conv_desc: cudnn::ConvolutionBiasActivation,
    filter: Tensor,
    offset: Tensor,
}

pub struct DenseBuilder {
    batch_size: i32,
    shape: [i32; 2],
    alpha: [f32; 2],
    act_desc: Option<cudnn::ActivationDescriptor>,
    filter: Option<Tensor>,
    offset: Option<Tensor>,
}

impl DenseBuilder {
    fn new(batch_size: i32, shape: [i32; 2]) -> Self {
        Self {
            batch_size: batch_size,
            shape: shape,
            alpha: [1.0, 0.0],
            act_desc: None,
            filter: None,
            offset: None
        }
    }

    pub fn with_alpha(mut self, alpha: [f32; 2]) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_tensors(mut self, tensors: &HashMap<String, Tensor>, name: &str) -> Self {
        self.filter = Some(tensors.get(&format!("{}:0", name)).cloned().expect("no filter available"));
        self.offset = Some(tensors.get(&format!("{}/offset:0", name)).cloned().expect("no offset available"));
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

    pub fn with_activation(mut self, act_desc: cudnn::ActivationDescriptor) -> Self {
        self.act_desc = Some(act_desc);
        self
    }

    fn create_transpose_filter(&self) -> Result<cudnn::Transform, cudnn::Status> {
        let num_outputs = self.shape[0];
        let num_inputs = self.shape[1];

        cudnn::Transform::new(
            cudnn::TensorDescriptor::new_ex(
                cudnn::DataType::Half,
                &[num_outputs, num_inputs, 1, 1],
                &[1, num_outputs, 1, 1],
            )?,
            cudnn::TensorDescriptor::new_ex(
                cudnn::DataType::Half,
                &[num_outputs, num_inputs, 1, 1],
                &[num_inputs, 1, 1, 1],
            )?,
            &[1.0, 0.0]
        )
    }

    fn create_activation_descriptor(&mut self) -> Result<cudnn::ActivationDescriptor, cudnn::Status> {
        if let Some(act_desc) = self.act_desc.take() {
            Ok(act_desc)
        } else {
            cudnn::ActivationDescriptor::relu()
        }
    }

    /// Returns a `ConvolutionDescriptor` for a one wide and high filter using a
    /// 32-bit compute type.
    fn create_convolution_descriptor(&self) -> Result<cudnn::ConvolutionDescriptor, cudnn::Status> {
        cudnn::ConvolutionDescriptor::new(
            &[0, 0],
            &[1, 1],
            &[1, 1],
            cudnn::ConvolutionMode::CrossCorrelation,
            cudnn::DataType::Float
        )
    }

    /// Returns a `FilterDescriptor` for a one wide and high filter for the given
    /// `num_outputs` and `num_inputs` features.
    /// 
    /// # Arguments
    /// 
    /// * `num_outputs` -
    /// * `num_inputs` -
    /// 
    fn create_filter_descriptor(&self) -> Result<cudnn::FilterDescriptor, cudnn::Status> {
        let num_outputs = self.shape[0];
        let num_inputs = self.shape[1];

        cudnn::FilterDescriptor::new(
            cudnn::DataType::Half,
            cudnn::TensorFormat::NCHW,
            &[num_outputs, num_inputs, 1, 1]
        )
    }

    fn create_dense_convolution_bias_activation(&mut self, handle: &cudnn::Handle) -> Result<cudnn::ConvolutionBiasActivation, cudnn::Status> {
        let num_outputs = self.shape[0];
        let num_inputs = self.shape[1];

        cudnn::ConvolutionBiasActivation::new(
            handle,
            self.alpha[0],
            create_dense_descriptor(self.batch_size, num_inputs)?,
            self.create_filter_descriptor()?,
            self.create_convolution_descriptor()?,
            self.alpha[1],
            create_offset_descriptor(num_outputs)?,
            self.create_activation_descriptor()?,
            create_dense_descriptor(self.batch_size, num_outputs)?,
        )
    }

    pub fn build(mut self, handle: &cudnn::Handle) -> Result<Dense, cudnn::Status> {
        Ok(Dense {
            conv_desc: self.create_dense_convolution_bias_activation(handle)?,
            transpose_filter: self.create_transpose_filter()?,
            filter: self.filter.take().expect("no filter given"),
            offset: self.offset.take().expect("no offset given")
        })
    }
}

impl Dense {
    pub fn new(batch_size: i32, shape: [i32; 2]) -> DenseBuilder {
        DenseBuilder::new(batch_size, shape)
    }

    pub fn offset(&self) -> &Tensor {
        &self.offset
    }

    pub fn prepare<A: cuda::Allocator + Clone>(
        &self,
        handle: &cudnn::Handle,
        allocator: &A,
        stream: &cuda::Stream
    ) -> Result<bool, Error> 
    {
        handle.set_stream(stream)?;
        if self.filter.copy_to_device(&stream)? && self.offset.copy_to_device(&stream)? {
            let temp = cuda::malloc(self.filter.get().size_in_bytes(), allocator)?;

            self.transpose_filter.forward(
                handle,
                self.filter.get().as_ptr(),
                temp.as_ptr()
            )?;
            self.filter.set_device_ptr(temp.unwrap());

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

        self.prepare(handle, allocator, stream)?;
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
    fn matmul_identity_8x8() {
        let filter = create_tensor(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let offset = create_tensor(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]);
        let handle = cudnn::Handle::new().unwrap();
        let matmul = Dense::new(1, [8, 8])
            .with_filter(filter.unwrap())
            .with_offset(offset.unwrap())
            .build(&handle)
            .unwrap();

        //
        let stream = cuda::Stream::default();
        let allocator = cuda::Native::default();
        let input = create_ptr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &allocator).unwrap();
        let output = matmul.forward(&handle, &input, &allocator, &stream).unwrap();

        assert_eq!(
            vec! [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            output.to_vec::<f16>(&stream).unwrap().iter().map(|&x| f32::from(x)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn matmul_identity_1x8() {
        let filter = create_tensor(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let offset = create_tensor(&[-36.0]);
        let handle = cudnn::Handle::new().unwrap();
        let matmul = Dense::new(1, [1, 8])
            .with_filter(filter.unwrap())
            .with_offset(offset.unwrap())
            .with_activation(cudnn::ActivationDescriptor::identity().unwrap())
            .build(&handle)
            .unwrap();

        //
        let stream = cuda::Stream::default();
        let allocator = cuda::Native::default();
        let input = create_ptr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &allocator).unwrap();
        let output = matmul.forward(&handle, &input, &allocator, &stream).unwrap();

        assert_eq!(
            vec! [0.0],
            output.to_vec::<f16>(&stream).unwrap().iter().map(|&x| f32::from(x)).collect::<Vec<_>>()
        );
    }
}
