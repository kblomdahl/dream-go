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

use libc::{c_void, c_int};

const ONE: f32 = 1.0;
const ZERO: f32 = 0.0;

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
    ) -> c_int;
}

#[derive(Debug)]
pub struct Softmax;

impl Softmax {
    pub fn new() -> Result<Softmax, Error> {
        Ok(Softmax)
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
            cudnnSoftmaxForward(
                handle.as_ptr(),
                cudnnSoftmaxAlgorithm_t::Accurate,
                cudnnSoftmaxMode_t::Channel,
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
}