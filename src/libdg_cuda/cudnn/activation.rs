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
    fn cudnnActivationForward(
        handle: cudnnHandle_t,
        activation_desc: cudnnActivationDescriptor_t,
        alpha: *const c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const c_void,
        beta: *const c_void,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut c_void
    ) -> cudnnStatus_t;
}

pub struct Activation {
    activation_desc: ActivationDescriptor,
    x: TensorDescriptor,
    y: TensorDescriptor,
    alpha: [f32; 2]
}

impl Activation {
    /// Returns an `Activation` that performs the given `activation_desc` and
    /// tensors.
    ///
    /// # Arguments
    ///
    /// * `activation_desc` -
    /// * `x` -
    /// * `y` -
    /// * `alpha` -
    ///
    pub fn new(
        activation_desc: ActivationDescriptor,
        x: TensorDescriptor,
        y: TensorDescriptor,
        alpha: [f32; 2]
    ) -> Result<Self, Status> {
        Ok(Self { activation_desc, x, y, alpha })
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
                cudnnActivationForward(
                    **handle,
                    *self.activation_desc,
                    &self.alpha[0] as *const _ as *const c_void,
                    *self.x,
                    x,
                    &self.alpha[1] as *const _ as *const c_void,
                    *self.y,
                    y
                )
            };

        status.into_result(())
    }
}

#[cfg(test)]
mod tests {
    // pass
}
