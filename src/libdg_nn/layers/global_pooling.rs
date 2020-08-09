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
use std::ptr;

use crate::layers::{create_tensor_descriptor, create_dense_descriptor};
use crate::Error;

pub struct GlobalPooling {
    reduce_tensor: cudnn2::ReduceTensor
}

pub struct GlobalPoolingBuilder {
    batch_size: i32,
    alpha: [f32; 2],
    num_channels: i32,
    reduce_op: cudnn2::ReduceTensorOp
}

impl GlobalPoolingBuilder {
    pub fn new(batch_size: i32, num_channels: i32, reduce_op: cudnn2::ReduceTensorOp) -> Self {
        Self {
            batch_size: batch_size,
            alpha: [1.0, 0.0],
            num_channels: num_channels,
            reduce_op: reduce_op
        }
    }

    fn create_reduce_tensor_descriptor(&self) -> Result<cudnn2::ReduceTensorDescriptor, cudnn2::Status> {
        cudnn2::ReduceTensorDescriptor::new(
            self.reduce_op,
            cudnn2::DataType::Float,
            cudnn2::NanPropagation::NotPropagateNaN,
            cudnn2::ReduceTensorIndices::NoIndices,
            cudnn2::IndicesType::_32
        )
    }

    fn create_reduce_tensor(&self) -> Result<cudnn2::ReduceTensor, cudnn2::Status> {
        cudnn2::ReduceTensor::new(
            self.create_reduce_tensor_descriptor()?,
            create_tensor_descriptor(self.batch_size, self.num_channels)?,
            create_dense_descriptor(self.batch_size, 1)?,
            self.alpha
        )
    }

    pub fn build(self) -> Result<GlobalPooling, cudnn2::Status> {
        Ok(GlobalPooling {
            reduce_tensor: self.create_reduce_tensor()?
        })
    }
}

impl GlobalPooling {
    pub fn new(batch_size: i32, num_channels: i32, reduce_op: cudnn2::ReduceTensorOp) -> GlobalPoolingBuilder {
        GlobalPoolingBuilder::new(batch_size, num_channels, reduce_op)
    }

    pub fn prepare(&self, handle: &cudnn2::Handle, stream: &cuda2::Stream) -> Result<(), Error> {
        handle.set_stream(stream)?;
        Ok(())
    }

    pub fn forward<A: cuda2::Allocator + Clone>(
        &self,
        handle: &cudnn2::Handle,
        input: &cuda2::SmartPtr<A>,
        allocator: &A,
        stream: &cuda2::Stream
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        let workspace = cuda2::malloc(self.reduce_tensor.size_in_bytes(handle)?, allocator)?;
        let output = cuda2::malloc(self.reduce_tensor.output().size_in_bytes()?, allocator)?;

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
