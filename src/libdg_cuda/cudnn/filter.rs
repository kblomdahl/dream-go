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

use libc::{c_int, c_void};
use std::ptr::{Unique, null_mut};
use Error;

#[link(name = "cudnn")]
extern {
    fn cudnnCreateFilterDescriptor(filter_desc: *mut cudnnFilterDescriptor_t) -> c_int;
    fn cudnnDestroyFilterDescriptor(filter_desc: cudnnFilterDescriptor_t) -> c_int;
    fn cudnnSetFilter4dDescriptor(
        filter_desc: cudnnFilterDescriptor_t,
        data_type: cudnnDataType_t,
        format: cudnnTensorFormat_t,
        k: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
    ) -> c_int;

    fn cudnnGetFilter4dDescriptor(
        filter_desc: cudnnFilterDescriptor_t,
        data_type: *mut cudnnDataType_t,
        format: *mut cudnnTensorFormat_t,
        k: *mut c_int,
        c: *mut c_int,
        h: *mut c_int,
        w: *mut c_int
    ) -> c_int;
}

#[derive(Debug)]
pub struct Filter(Unique<c_void>);

impl Drop for Filter {
    fn drop(&mut self) {
        unsafe { cudnnDestroyFilterDescriptor(self.as_ptr()) };
    }
}

impl Filter {
    pub fn new(
        data_type: cudnnDataType_t,
        format: cudnnTensorFormat_t,
        dims: &[usize]
    ) -> Result<Filter, Error>
    {
        let mut filter_desc = null_mut();
        let success = unsafe { cudnnCreateFilterDescriptor(&mut filter_desc) };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        let success = unsafe {
            cudnnSetFilter4dDescriptor(
                filter_desc,
                data_type,
                format,
                dims[0] as i32,
                dims[1] as i32,
                dims[2] as i32,
                dims[3] as i32
            )
        };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        Ok(Filter(Unique::new(filter_desc).unwrap()))
    }

    pub fn dims(&self) -> Result<(usize, usize, usize, usize), Error> {
        let (mut k, mut c, mut h, mut w) = (0, 0, 0, 0);
        let mut data_type = cudnnDataType_t::Float;
        let mut format = cudnnTensorFormat_t::NCHW;
        let success = unsafe {
            cudnnGetFilter4dDescriptor(
                self.as_ptr(),
                &mut data_type,
                &mut format,
                &mut k,
                &mut c,
                &mut h,
                &mut w
            )
        };

        if success != 0 {
            Err(Error::CudnnError(success))
        } else {
            Ok((k as usize, c as usize, h as usize, w as usize))
        }
    }

    pub(super) fn as_ptr(&self) -> cudnnFilterDescriptor_t {
        self.0.as_ptr()
    }
}
