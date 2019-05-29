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

use std::ptr::{null_mut, Unique};
use libc::c_void;

const ONE: f32 = 1.0;
const ZERO: f32 = 0.0;

#[link(name = "cudnn")]
extern {
    fn cudnnCreatePoolingDescriptor(pooling_desc: *mut cudnnPoolingDescriptor_t) -> c_int;
    fn cudnnDestroyPoolingDescriptor(pooling_desc: cudnnPoolingDescriptor_t) -> c_int;

    fn cudnnSetPooling2dDescriptor(
        pooling_desc: cudnnPoolingDescriptor_t,
        mode: cudnnPoolingMode_t,
        max_pooling_nan_opt: cudnnNanPropagation_t,
        window_height: c_int,
        window_width: c_int,
        vertical_padding: c_int,
        horizontal_padding: c_int,
        vertical_stride: c_int,
        horizontal_stride: c_int
    ) -> c_int;

    fn cudnnPoolingForward(
        handle: cudnnHandle_t,
        pooling_desc: cudnnPoolingDescriptor_t,
        alpha: *const c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const c_void,
        beta: *const c_void,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut c_void
    ) -> c_int;
}

#[derive(Debug)]
pub struct Pooling(Unique<c_void>);

impl Drop for Pooling {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyPoolingDescriptor(self.as_ptr());
        }
    }
}

impl Pooling {
    pub fn new(
        mode: cudnnPoolingMode_t,
        max_pooling_nan_opt: cudnnNanPropagation_t,
        window: (usize, usize),
        padding: (usize, usize),
        strides: (usize, usize)
    ) -> Result<Pooling, Error>
    {
        let mut pooling_desc = null_mut();
        let success = unsafe { cudnnCreatePoolingDescriptor(&mut pooling_desc) };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        let success = unsafe {
            cudnnSetPooling2dDescriptor(
                pooling_desc,
                mode,
                max_pooling_nan_opt,
                window.0 as c_int,
                window.1 as c_int,
                padding.0 as c_int,
                padding.1 as c_int,
                strides.0 as c_int,
                strides.1 as c_int
            )
        };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        Ok(Pooling(Unique::new(pooling_desc).unwrap()))
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
        let success = unsafe {
            cudnnPoolingForward(
                handle.as_ptr(),
                self.as_ptr(),
                &ONE as *const _ as *const c_void,
                x.as_ptr(), x_data,
                &ZERO as *const _ as *const c_void,
                y.as_ptr(), y_data
            )
        };

        if success != 0 {
            Err(Error::CudnnError(success))
        } else {
            Ok(())
        }
    }

    pub(super) fn as_ptr(&self) -> cudnnPoolingDescriptor_t {
        self.0.as_ptr()
    }
}