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
use libc::{c_void, c_int};

#[allow(non_camel_case_types)]
pub type cudnnFilterDescriptor_t = *const c_void;

#[link(name = "cudnn")]
extern {
    fn cudnnCreateFilterDescriptor(filter_desc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;
    fn cudnnDestroyFilterDescriptor(filter_desc: cudnnFilterDescriptor_t) -> cudnnStatus_t;
    fn cudnnSetFilter4dDescriptor(
        filter_desc: cudnnFilterDescriptor_t,
        data_type: cudnnDataType_t,
        format: cudnnTensorFormat_t,
        k: c_int,
        c: c_int,
        r: c_int,
        s: c_int,
    ) -> cudnnStatus_t;
    fn cudnnGetFilter4dDescriptor(
        filter_desc: cudnnFilterDescriptor_t,
        data_type: *mut cudnnDataType_t,
        format: *mut cudnnTensorFormat_t,
        k: *mut c_int,
        c: *mut c_int,
        r: *mut c_int,
        s: *mut c_int,
    ) -> cudnnStatus_t;
}

struct GetFilter4dDescriptor {
    data_type: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    k: c_int,
    c: c_int,
    r: c_int,
    s: c_int
}

impl GetFilter4dDescriptor {
    fn new(filter_desc: cudnnFilterDescriptor_t) -> Result<GetFilter4dDescriptor, Status> {
        let mut out = Self {
            data_type: cudnnDataType_t::Float,
            format: cudnnTensorFormat_t::NCHW,
            k: 0,
            c: 0,
            r: 0,
            s: 0
        };
        let status =
            unsafe {
                cudnnGetFilter4dDescriptor(
                    filter_desc,
                    &mut out.data_type,
                    &mut out.format,
                    &mut out.k,
                    &mut out.c,
                    &mut out.r,
                    &mut out.s
                )
            };

        status.into_result(out)
    }
}

pub struct FilterDescriptor {
    filter_desc: cudnnFilterDescriptor_t
}

impl Deref for FilterDescriptor {
    type Target = cudnnFilterDescriptor_t;

    fn deref(&self) -> &Self::Target {
        &self.filter_desc
    }
}

impl Drop for FilterDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyFilterDescriptor(self.filter_desc) };
    }
}

impl FilterDescriptor {
    pub fn empty() -> Result<Self, Status> {
        let mut out = Self { filter_desc: ptr::null_mut() };
        let status = unsafe { cudnnCreateFilterDescriptor(&mut out.filter_desc) };

        status.into_result(out)
    }

    /// This function creates a filter descriptor object by allocating the
    /// memory needed to hold its opaque structure. For more information, see
    /// `cudnnFilterDescriptor_t`.
    /// 
    /// * `K` is the number of output feature maps
    /// * `C` is the number of input feature maps
    /// * `R` is the number of rows per filter
    /// * `S` is the number of columns per filter
    /// 
    /// # Arguments
    /// 
    /// * `data_type` -
    /// * `format` - Order of the data in the filter, `NCHW` is `KCRS`, and
    ///              `NHWC` is `KRSC`.
    /// * `shape` - Shape in KCHW order.
    /// 
    pub fn new(
        data_type: DataType,
        format: TensorFormat,
        shape: &[i32]
    ) -> Result<Self, Status>
    {
        debug_assert_eq!(shape.len(), 4);

        Self::empty().and_then(|out| {
            let status =
                unsafe {
                    cudnnSetFilter4dDescriptor(
                        out.filter_desc,
                        data_type,
                        format,
                        shape[0],
                        shape[1],
                        shape[2],
                        shape[3]
                    )
                };

            status.into_result(out)
        })
    }

    pub fn data_type(&self) -> Result<DataType, Status> {
        GetFilter4dDescriptor::new(self.filter_desc).map(|out| out.data_type)
    }

    pub fn format(&self) -> Result<TensorFormat, Status> {
        GetFilter4dDescriptor::new(self.filter_desc).map(|out| out.format)
    }

    pub fn shape(&self) -> Result<Vec<i32>, Status> {
        GetFilter4dDescriptor::new(self.filter_desc).map(
            |out| vec! [out.k, out.c, out.r, out.s]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create() {
        let filter_desc = FilterDescriptor::new(
            DataType::Half,
            TensorFormat::NHWC,
            &vec! [128, 128, 3, 3]
        );

        assert!(filter_desc.is_ok());
    }

    #[test]
    fn get_tensor_desc() {
        let filter_desc = FilterDescriptor::new(
            DataType::Half,
            TensorFormat::NHWC,
            &vec! [128, 32, 5, 5]
        ).unwrap();

        assert_eq!(filter_desc.data_type(), Ok(DataType::Half));
        assert_eq!(filter_desc.format(), Ok(TensorFormat::NHWC));
        assert_eq!(filter_desc.shape(), Ok(vec! [128, 32, 5, 5]));
    }
}
