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
    fn cudnnAddTensor(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        a_desc: cudnnTensorDescriptor_t,
        a: *const c_void,
        beta: *const c_void,
        c_desc: cudnnTensorDescriptor_t,
        c: *mut c_void,
    ) -> cudnnStatus_t;
}

pub struct Add {
    a: TensorDescriptor,
    c: TensorDescriptor,
    alpha: [f32; 2]
}

impl Add {
    pub fn new(
        a: TensorDescriptor,
        c: TensorDescriptor,
        alpha: [f32; 2]
    ) -> Result<Self, Status>
    {
        Ok(Self { a, c, alpha })
    }

    pub fn forward(
        &self,
        handle: &Handle,
        a: *const c_void,
        c: *mut c_void
    ) -> Result<(), Status>
    {
        let status =
            unsafe {
                cudnnAddTensor(
                    **handle,
                    &self.alpha[0] as *const _ as *const c_void,
                    *self.a,
                    a,
                    &self.alpha[1] as *const _ as *const c_void,
                    *self.c,
                    c
                )
            };

        status.into_result(())
    }
}

#[cfg(test)]
mod tests {
    // pass
}
