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
use libc::{c_void, c_int, size_t};

#[allow(non_camel_case_types)]
pub type cudnnTensorDescriptor_t = *const c_void;

#[link(name = "cudnn_ops_infer")]
extern {
    fn cudnnCreateTensorDescriptor(tensor_desc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;
    fn cudnnDestroyTensorDescriptor(tensor_desc: cudnnTensorDescriptor_t) -> cudnnStatus_t;
    fn cudnnSetTensor4dDescriptor(
        tensor_desc: cudnnTensorDescriptor_t,
        format: cudnnTensorFormat_t,
        data_type: cudnnDataType_t,
        n: c_int,
        c: c_int,
        h: c_int,
        w: c_int
    ) -> cudnnStatus_t;
    fn cudnnSetTensor4dDescriptorEx(
        tensor_desc: cudnnTensorDescriptor_t,
        data_type: cudnnDataType_t,
        n: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
        n_stride: c_int,
        c_stride: c_int,
        h_stride: c_int,
        w_stride: c_int
    ) -> cudnnStatus_t;
    fn cudnnGetTensor4dDescriptor(
        tensor_desc: cudnnTensorDescriptor_t,
        data_type: *mut cudnnDataType_t,
        n: *mut c_int,
        c: *mut c_int,
        h: *mut c_int,
        w: *mut c_int,
        n_stride: *mut c_int,
        c_stride: *mut c_int,
        h_stride: *mut c_int,
        w_stride: *mut c_int,
    ) -> cudnnStatus_t;
    fn cudnnGetTensorSizeInBytes(tensor_desc: cudnnTensorDescriptor_t, size: *mut size_t) -> cudnnStatus_t;
}

struct GetTensor4dDescriptor {
    data_type: cudnnDataType_t,
    n: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
    n_stride: c_int,
    c_stride: c_int,
    h_stride: c_int,
    w_stride: c_int,
}

impl GetTensor4dDescriptor {
    fn new(tensor_desc: cudnnTensorDescriptor_t) -> Result<Self, Status> {
        let mut out = Self {
            data_type: cudnnDataType_t::Float,
            n: 0,
            c: 0,
            h: 0,
            w: 0,
            n_stride: 0,
            c_stride: 0,
            h_stride: 0,
            w_stride: 0,
        };
        let status  = unsafe {
            cudnnGetTensor4dDescriptor(
                tensor_desc,
                &mut out.data_type,
                &mut out.n,
                &mut out.c,
                &mut out.h,
                &mut out.w,
                &mut out.n_stride,
                &mut out.c_stride,
                &mut out.h_stride,
                &mut out.w_stride,
            )
        };

        status.into_result(out)
    }
}

/// `cudnnCreateTensorDescriptor_t` is a pointer to an opaque structure holding
/// the description of a generic n-D dataset. `cudnnCreateTensorDescriptor()` is
/// used to create one instance, and one of the routines
/// `cudnnSetTensorNdDescriptor()`, `cudnnSetTensor4dDescriptor()` or
/// `cudnnSetTensor4dDescriptorEx()` must be used to initialize this instance.
pub struct TensorDescriptor {
    tensor_desc: cudnnTensorDescriptor_t
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyTensorDescriptor(self.tensor_desc) };
    }
}

impl Deref for TensorDescriptor {
    type Target = cudnnTensorDescriptor_t;

    fn deref(&self) -> &Self::Target {
        &self.tensor_desc
    }
}

impl TensorDescriptor {
    pub fn empty() -> Result<Self, Status> {
        let mut out = Self { tensor_desc: ptr::null_mut() };
        let status = unsafe { cudnnCreateTensorDescriptor(&mut out.tensor_desc) };

        status.into_result(out)
    }

    /// Returns a tensor descriptor created by `cudnnSetTensor4dDescriptor()`
    /// with the given `format`, `data_type`, and `shape`.
    ///
    /// # Arguments
    ///
    /// * `format` -
    /// * `data_type` -
    /// * `shape` - Shape in NCHW order.
    ///
    pub fn new(
        format: TensorFormat,
        data_type: DataType,
        shape: [i32; 4],
    ) -> Result<Self, Status>
    {
        debug_assert_eq!(shape.len(), 4);

        Self::empty().and_then(|out| {
            let status =
                unsafe {
                    cudnnSetTensor4dDescriptor(
                        out.tensor_desc,
                        format,
                        data_type,
                        shape[0],
                        shape[1],
                        shape[2],
                        shape[3]
                    )
                };

            status.into_result(out)
        })
    }

    /// Returns a tensor descriptor created by `cudnnSetTensor4dDescriptorEx()`
    /// with the given `format`, `data_type`, `shape`, and `stride`.
    ///
    /// # Arguments
    ///
    /// * `data_type` -
    /// * `shape` - Shape in NCHW order.
    /// * `stride` -
    ///
    pub fn new_ex(
        data_type: DataType,
        shape: [i32; 4],
        stride: [i32; 4],
    ) -> Result<Self, Status>
    {
        debug_assert_eq!(shape.len(), 4);
        debug_assert_eq!(stride.len(), 4);

        Self::empty().and_then(|out| {
            let status =
                unsafe {
                    cudnnSetTensor4dDescriptorEx(
                        out.tensor_desc,
                        data_type,
                        shape[0],
                        shape[1],
                        shape[2],
                        shape[3],
                        stride[0],
                        stride[1],
                        stride[2],
                        stride[3],
                    )
                };

            status.into_result(out)
        })
    }

    pub fn data_type(&self) -> Result<DataType, Status> {
        GetTensor4dDescriptor::new(self.tensor_desc).map(|out| out.data_type)
    }

    pub fn shape(&self) -> Result<[i32; 4], Status> {
        GetTensor4dDescriptor::new(self.tensor_desc).map(
            |out| [out.n, out.c, out.h, out.w]
        )
    }

    pub fn stride(&self) -> Result<[i32; 4], Status> {
        GetTensor4dDescriptor::new(self.tensor_desc).map(
            |out| [out.n_stride, out.c_stride, out.h_stride, out.w_stride]
        )
    }

    pub fn size_in_bytes(&self) -> Result<usize, Status> {
        let mut size_in_bytes = 0;
        let status = unsafe { cudnnGetTensorSizeInBytes(self.tensor_desc, &mut size_in_bytes) };

        status.into_result(size_in_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_tensor_desc() {
        let tensor_desc = TensorDescriptor::new(
            TensorFormat::NHWC,
            DataType::Float,
            [1, 1, 1, 1]
        );

        assert!(tensor_desc.is_ok());
    }

    #[test]
    fn get_tensor_desc_nhwc() {
        let tensor_desc = TensorDescriptor::new(
            TensorFormat::NHWC,
            DataType::Float,
            [1, 2, 3, 4]
        ).unwrap();

        assert_eq!(tensor_desc.data_type(), Ok(DataType::Float));
        assert_eq!(tensor_desc.shape(), Ok([1, 2, 3, 4]));
        assert_eq!(tensor_desc.stride(), Ok([24, 1, 8, 2]));
    }

    #[test]
    fn get_tensor_desc_nchw() {
        let tensor_desc = TensorDescriptor::new(
            TensorFormat::NCHW,
            DataType::Float,
            [1, 2, 3, 4]
        ).unwrap();

        assert_eq!(tensor_desc.data_type(), Ok(DataType::Float));
        assert_eq!(tensor_desc.shape(), Ok([1, 2, 3, 4]));
        assert_eq!(tensor_desc.stride(), Ok([24, 12, 4, 1]));
    }

    #[test]
    fn size_in_bytes() {
        let tensor_desc = TensorDescriptor::new(
            TensorFormat::NCHW,
            DataType::Half,
            [1, 2, 3, 4]
        ).unwrap();

        assert_eq!(tensor_desc.size_in_bytes(), Ok(48));
    }
}
