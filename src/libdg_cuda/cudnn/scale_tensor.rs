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

use libc::{c_void};

#[link(name = "cudnn_ops_infer")]
extern {
    fn cudnnScaleTensor(
        handle: cudnnHandle_t,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut c_void,
        alpha: *const c_void
    ) -> cudnnStatus_t;
}

pub struct Scale {
    y: TensorDescriptor,
    alpha: f32
}

impl Scale {
    pub fn new(
        y: TensorDescriptor,
        alpha: f32
    ) -> Result<Self, Status>
    {
        Ok(Self { y, alpha })
    }

    pub fn forward(
        &self,
        handle: &Handle,
        y: *mut c_void
    ) -> Result<(), Status>
    {
        let status =
            unsafe {
                cudnnScaleTensor(
                    **handle,
                    *self.y,
                    y,
                    &self.alpha as *const _ as *const c_void
                )
            };

        status.into_result(())
    }
}

#[cfg(test)]
mod tests {
    // pass
}
