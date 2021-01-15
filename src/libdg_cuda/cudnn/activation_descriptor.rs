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

use std::ptr;
use std::ops::Deref;
use libc::{c_void, c_double};

#[allow(non_camel_case_types)]
pub type cudnnActivationDescriptor_t = *const c_void;

#[link(name = "cudnn_ops_infer")]
extern {
    fn cudnnCreateActivationDescriptor(activation_desc: *mut cudnnActivationDescriptor_t) -> cudnnStatus_t;
    fn cudnnDestroyActivationDescriptor(activation_desc: cudnnActivationDescriptor_t) -> cudnnStatus_t;
    fn cudnnSetActivationDescriptor(
        activation_desc: cudnnActivationDescriptor_t,
        mode: cudnnActivationMode_t,
        relu_nan_opt: cudnnNanPropagation_t,
        coef: c_double
    ) -> cudnnStatus_t;
    fn cudnnGetActivationDescriptor(
        activation_desc: cudnnActivationDescriptor_t,
        mode: *mut cudnnActivationMode_t,
        relu_nan_opt: *mut cudnnNanPropagation_t,
        coef: *mut c_double
    ) -> cudnnStatus_t;
}

struct GetActivationDescriptor {
    mode: cudnnActivationMode_t,
    relu_nan_opt: cudnnNanPropagation_t,
    coef: c_double
}

impl GetActivationDescriptor {
    fn new(activation_desc: cudnnActivationDescriptor_t) -> Result<Self, Status> {
        let mut out = Self {
            mode: cudnnActivationMode_t::Sigmoid,
            relu_nan_opt: cudnnNanPropagation_t::NotPropagateNaN,
            coef: 0.0
        };
        let status =
            unsafe {
                cudnnGetActivationDescriptor(
                    activation_desc,
                    &mut out.mode,
                    &mut out.relu_nan_opt,
                    &mut out.coef
                )
            };

        status.into_result(out)
    }
}

pub struct ActivationDescriptor {
    act_desc: cudnnActivationDescriptor_t
}

impl Deref for ActivationDescriptor {
    type Target = cudnnActivationDescriptor_t;

    fn deref(&self) -> &Self::Target {
        &self.act_desc
    }
}

impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyActivationDescriptor(self.act_desc) };
    }
}

impl ActivationDescriptor {
    pub fn empty() -> Result<Self, Status> {
        let mut out = Self { act_desc: ptr::null_mut() };
        let status = unsafe { cudnnCreateActivationDescriptor(&mut out.act_desc) };

        status.into_result(out)
    }

    pub fn new(
        mode: ActivationMode,
        relu_nan_op: NanPropagation,
        coef: f64
    ) -> Result<Self, Status>
    {
        Self::empty().and_then(|out| {
            let status = unsafe {
                cudnnSetActivationDescriptor(
                    out.act_desc,
                    mode,
                    relu_nan_op,
                    coef
                )
            };

            status.into_result(out)
        })
    }

    pub fn relu() -> Result<Self, Status> {
        Self::new(
            ActivationMode::Relu,
            NanPropagation::NotPropagateNaN,
            0.0
        )
    }

    pub fn tanh() -> Result<Self, Status> {
        Self::new(
            ActivationMode::Tanh,
            NanPropagation::NotPropagateNaN,
            0.0
        )
    }

    pub fn identity() -> Result<Self, Status> {
        Self::new(
            ActivationMode::Identity,
            NanPropagation::NotPropagateNaN,
            0.0
        )
    }

    pub fn coef(&self) -> Result<f64, Status> {
        GetActivationDescriptor::new(self.act_desc).map(|out| out.coef)
    }

    pub fn mode(&self) -> Result<ActivationMode, Status> {
        GetActivationDescriptor::new(self.act_desc).map(|out| out.mode)
    }

    pub fn relu_nan_opt(&self) -> Result<NanPropagation, Status> {
        GetActivationDescriptor::new(self.act_desc).map(|out| out.relu_nan_opt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create() {
        let activation_desc = ActivationDescriptor::new(
            ActivationMode::Relu,
            NanPropagation::NotPropagateNaN,
            0.0
        );

        assert!(activation_desc.is_ok());
    }

    #[test]
    fn get_activation_descriptor() {
        let activation_desc = ActivationDescriptor::new(
            ActivationMode::ClippedRelu,
            NanPropagation::PropagateNaN,
            6.0
        ).unwrap();

        assert_eq!(activation_desc.mode(), Ok(ActivationMode::ClippedRelu));
        assert_eq!(activation_desc.relu_nan_opt(), Ok(NanPropagation::PropagateNaN));
        assert_eq!(activation_desc.coef(), Ok(6.0));
    }
}
