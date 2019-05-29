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

use libc::{c_int, c_void};

use ::{copy_nonoverlapping, Error};
use cudaMemcpyKind_t;

use super::*;

extern {
    fn cudnnScaleTensor(
        handle: cudnnHandle_t,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut c_void,
        alpha: *const c_void
    ) -> c_int;
}

#[derive(Debug)]
pub struct Scale {
    alpha: f32
}

impl Scale {
    pub fn new(alpha: f32) -> Result<Scale, Error> {
        Ok(Scale {
            alpha
        })
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
        let stream = handle.get_stream()?;

        copy_nonoverlapping(
            x_data,
            y_data,
            x.size_in_bytes()?,
            cudaMemcpyKind_t::DeviceToDevice,
            &stream
        )?;

        let success = unsafe {
            cudnnScaleTensor(
                handle.as_ptr(),
                y.as_ptr(), y_data,
                &self.alpha as *const _ as *const c_void
            )
        };

        if success != 0 {
            Err(Error::CudnnError(success))
        } else {
            Ok(())
        }
    }
}