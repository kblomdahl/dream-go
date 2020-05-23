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

#[link(name = "cudnn")]
extern {
    fn cudnnSoftmaxForward(
        handle: cudnnHandle_t,
        algorithm: cudnnSoftmaxAlgorithm_t,
        mode: cudnnSoftmaxMode_t,
        alpha: *const c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const c_void,
        beta: *const c_void,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut c_void
    ) -> cudnnStatus_t;
}

pub struct Softmax {
    mode: SoftmaxMode,
    x: TensorDescriptor,
    y: TensorDescriptor,
    alpha: [f32; 2]
}

impl Softmax {
    pub fn new(
        mode: SoftmaxMode,
        x: TensorDescriptor,
        y: TensorDescriptor,
        alpha: &[f32]
    ) -> Result<Self, Status>
    {
        let alpha = [alpha[0], alpha[1]];

        Ok(Self { mode, x, y, alpha })
    }

    pub fn forward(
        &self,
        handle: &Handle,
        x: *const c_void,
        y: *mut c_void
    ) -> Result<(), Status>
    {
        let status =
            unsafe {
                cudnnSoftmaxForward(
                    **handle,
                    cudnnSoftmaxAlgorithm_t::Accurate,
                    self.mode,
                    &self.alpha[0] as *const _ as *const c_void,
                    *self.x, x,
                    &self.alpha[1] as *const _ as *const c_void,
                    *self.y, y
                )
            };

        status.into_result(())
    }
}

#[cfg(test)]
mod tests {
    // pass
}

