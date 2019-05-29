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

use super::*;
use Error;

use libc::{c_void, c_int, c_double};
use std::ptr::{null_mut, Unique};

#[link(name = "cudnn")]
extern {
    fn cudnnCreateActivationDescriptor(activation_desc: *mut cudnnActivationDescriptor_t) -> c_int;
    fn cudnnDestroyActivationDescriptor(activation_desc: cudnnActivationDescriptor_t) -> c_int;
    fn cudnnGetActivationDescriptor(
        activation_desc: cudnnActivationDescriptor_t,
        mode: *mut cudnnActivationMode_t,
        relu_nan_opt: *mut cudnnNanPropagation_t,
        coef: *mut c_double
    ) -> c_int;

    fn cudnnSetActivationDescriptor(
        activation_desc: cudnnActivationDescriptor_t,
        mode: cudnnActivationMode_t,
        relu_nan_opt: cudnnNanPropagation_t,
        coef: c_double
    ) -> c_int;

    fn cudnnActivationForward(
        handle: cudnnHandle_t,
        activation_desc: cudnnActivationDescriptor_t,
        alpha: *const c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const c_void,
        beta: *const c_void,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut c_void
    ) -> c_int;
}

#[derive(Debug)]
pub struct Activation(Unique<c_void>);

impl Drop for Activation {
    fn drop(&mut self) {
        unsafe { cudnnDestroyActivationDescriptor(self.as_ptr()) };
    }
}

impl Activation {
    pub fn new(
        mode: cudnnActivationMode_t,
        relu_nan_opt: cudnnNanPropagation_t,
        coef: f64
    ) -> Result<Activation, Error>
    {
        let mut activation_desc = null_mut();
        let success = unsafe { cudnnCreateActivationDescriptor(&mut activation_desc) };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        let success = unsafe {
            cudnnSetActivationDescriptor(
                activation_desc,
                mode,
                relu_nan_opt,
                coef as c_double
            )
        };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        Ok(Activation(Unique::new(activation_desc).unwrap()))
    }

    pub fn info(&self) -> Result<(cudnnActivationMode_t, cudnnNanPropagation_t, f64), Error> {
        let mut activation_mode = cudnnActivationMode_t::Identity;
        let mut relu_nan_opt = cudnnNanPropagation_t::NotPropagateNaN;
        let mut coef = 0.0;
        let success = unsafe {
            cudnnGetActivationDescriptor(
                self.as_ptr(),
                &mut activation_mode as *mut _,
                &mut relu_nan_opt as *mut _,
                &mut coef as *mut _,
            )
        };

        if success != 0 {
            Err(Error::CudnnError(success))
        } else {
            Ok((activation_mode, relu_nan_opt, coef))
        }
    }

    pub fn forward(
        &self,
        handle: &Handle,
        x: &Tensor,
        x_data: *const c_void,
        y: &Tensor,
        y_data: *mut c_void
    ) -> Result<(), Error>
    {
        const ONE: f32 = 1.0;
        const ZERO: f32 = 0.0;

        let success = unsafe {
            cudnnActivationForward(
                handle.as_ptr(),
                self.as_ptr(),
                &ONE as *const _ as *const c_void,
                x.as_ptr(),
                x_data,
                &ZERO as *const _ as *const c_void,
                y.as_ptr(),
                y_data
            )
        };

        if success != 0 {
            Err(Error::CudnnError(success))
        } else {
            Ok(())
        }
    }

    pub(super) fn as_ptr(&self) -> cudnnActivationDescriptor_t {
        self.0.as_ptr()
    }
}
