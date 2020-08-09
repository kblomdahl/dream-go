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
use std::ptr;

use crate::layers::{create_tensor_descriptor, create_dense_descriptor};
use crate::Error;

pub struct GlobalPooling {
    reduce_tensor: cudnn::ReduceTensor
}

pub struct GlobalPoolingBuilder {
    batch_size: i32,
    alpha: [f32; 2],
    num_channels: i32,
    reduce_op: cudnn::ReduceTensorOp
}

impl GlobalPoolingBuilder {
    pub fn new(batch_size: i32, num_channels: i32, reduce_op: cudnn::ReduceTensorOp) -> Self {
        Self {
            batch_size: batch_size,
            alpha: [1.0, 0.0],
            num_channels: num_channels,
            reduce_op: reduce_op
        }
    }

    fn create_reduce_tensor_descriptor(&self) -> Result<cudnn::ReduceTensorDescriptor, cudnn::Status> {
        cudnn::ReduceTensorDescriptor::new(
            self.reduce_op,
            cudnn::DataType::Float,
            cudnn::NanPropagation::NotPropagateNaN,
            cudnn::ReduceTensorIndices::NoIndices,
            cudnn::IndicesType::_32
        )
    }

    fn create_reduce_tensor(&self) -> Result<cudnn::ReduceTensor, cudnn::Status> {
        cudnn::ReduceTensor::new(
            self.create_reduce_tensor_descriptor()?,
            create_tensor_descriptor(self.batch_size, self.num_channels)?,
            create_dense_descriptor(self.batch_size, 1)?,
            self.alpha
        )
    }

    pub fn build(self) -> Result<GlobalPooling, cudnn::Status> {
        Ok(GlobalPooling {
            reduce_tensor: self.create_reduce_tensor()?
        })
    }
}

impl GlobalPooling {
    pub fn new(batch_size: i32, num_channels: i32, reduce_op: cudnn::ReduceTensorOp) -> GlobalPoolingBuilder {
        GlobalPoolingBuilder::new(batch_size, num_channels, reduce_op)
    }

    pub fn prepare(&self, handle: &cudnn::Handle, stream: &cuda::Stream) -> Result<(), Error> {
        handle.set_stream(stream)?;
        Ok(())
    }

    pub fn forward<A: cuda::Allocator + Clone>(
        &self,
        handle: &cudnn::Handle,
        input: &cuda::SmartPtr<A>,
        allocator: &A,
        stream: &cuda::Stream
    ) -> Result<cuda::SmartPtr<A>, Error>
    {
        let workspace = cuda::malloc(self.reduce_tensor.size_in_bytes(handle)?, allocator)?;
        let output = cuda::malloc(self.reduce_tensor.output().size_in_bytes()?, allocator)?;

        self.prepare(handle, stream)?;
        self.reduce_tensor.forward(
            handle,
            ptr::null_mut(), 0,
            workspace.as_ptr(), workspace.size_in_bytes(),
            input.as_ptr(),
            output.as_ptr()
        )?;

        Ok(output)
    }
}
