// Copyright 2020 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use crate::cudnn::*;

use libc::{c_void, size_t};

#[link(name = "cudnn")]
extern {
    fn cudnnReduceTensor(
        handle: cudnnHandle_t,
        reduce_tensor_desc: cudnnReduceTensorDescriptor_t,
        indices: *mut c_void,
        indices_size_in_bytes: size_t,
        workspace: *mut c_void,
        workspace_size_in_bytes: size_t,
        alpha: *const c_void,
        a_desc: cudnnTensorDescriptor_t,
        a: *const c_void,
        beta: *const c_void,
        c_desc: cudnnTensorDescriptor_t,
        c: *mut c_void
    ) -> cudnnStatus_t;

    fn cudnnGetReductionWorkspaceSize(
        handle: cudnnHandle_t,
        reduce_reduce_desc: cudnnReduceTensorDescriptor_t,
        a_desc: cudnnTensorDescriptor_t,
        b_desc: cudnnTensorDescriptor_t,
        size_in_bytes: *mut size_t
    ) -> cudnnStatus_t;
}

pub struct ReduceTensor {
    reduce_tensor_desc: ReduceTensorDescriptor,
    x: TensorDescriptor,
    y: TensorDescriptor,
    alpha: [f32; 2],
}

impl ReduceTensor {
    pub fn new(
        reduce_tensor_desc: ReduceTensorDescriptor,
        x: TensorDescriptor,
        y: TensorDescriptor,
        alpha: [f32; 2]
    ) -> Result<Self, Status>
    {
        Ok(Self { reduce_tensor_desc, x, y, alpha })
    }

    pub fn forward(
        &self,
        handle: &Handle,
        indices: *mut c_void,
        indices_size_in_bytes: size_t,
        workspace: *mut c_void,
        workspace_size_in_bytes: size_t,
        x: *const c_void,
        y: *mut c_void
    ) -> Result<(), Status>
    {
        let status =
            unsafe {
                cudnnReduceTensor(
                    **handle,
                    *self.reduce_tensor_desc,
                    indices, indices_size_in_bytes,
                    workspace, workspace_size_in_bytes,
                    &self.alpha[0] as *const _ as *const c_void,
                    *self.x, x,
                    &self.alpha[1] as *const _ as *const c_void,
                    *self.y, y,
                )
            };

        status.into_result(())
    }

    pub fn output(&self) -> &TensorDescriptor {
        &self.y
    }

    pub fn size_in_bytes(&self, handle: &Handle) -> Result<usize, Status> {
        let mut out = 0;
        let status = unsafe {
            cudnnGetReductionWorkspaceSize(
                **handle,
                *self.reduce_tensor_desc,
                *self.x,
                *self.y,
                &mut out
            )
        };

        status.into_result(out)
    }
}

#[cfg(test)]
mod tests {
    // pass
}
