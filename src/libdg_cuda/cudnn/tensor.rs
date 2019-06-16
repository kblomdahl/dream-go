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
use ::{Error, copy_nonoverlapping};

use libc::{c_void, c_int};
use std::ptr::{Unique, null_mut};
use ::{cudaMemcpyKind_t, Stream};
use Ptr;
use dg_utils::types::f16;

#[link(name = "cudnn")]
extern {
    fn cudnnCreateTensorDescriptor(tensor_desc: *mut cudnnTensorDescriptor_t) -> c_int;
    fn cudnnDestroyTensorDescriptor(tensor_desc: cudnnTensorDescriptor_t) -> c_int;
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
    ) -> c_int;

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
        w_stride: *mut c_int
    ) -> c_int;
}

#[derive(Clone, Debug)]
pub struct Tensor(Unique<c_void>);

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyTensorDescriptor(self.as_ptr()) };
    }
}

impl Tensor {
    fn new(
        data_type: cudnnDataType_t,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        n_stride: usize,
        c_stride: usize,
        h_stride: usize,
        w_stride: usize
    ) -> Result<Tensor, Error>
    {
        let mut tensor_desc = null_mut();
        let success = unsafe { cudnnCreateTensorDescriptor(&mut tensor_desc) };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        let success = unsafe {
            cudnnSetTensor4dDescriptorEx(
                tensor_desc,
                data_type,
                n as c_int,
                c as c_int,
                h as c_int,
                w as c_int,
                n_stride as c_int,
                c_stride as c_int,
                h_stride as c_int,
                w_stride as c_int
            )
        };

        if success != 0 {
            return Err(Error::CudnnError(success));
        }

        Ok(Tensor(Unique::new(tensor_desc).unwrap()))
    }

    pub(super) fn as_ptr(&self) -> cudnnTensorDescriptor_t {
        self.0.as_ptr()
    }

    pub fn from_nchw(data_type: cudnnDataType_t, dims: &[usize]) -> Result<Tensor, Error> {
        Tensor::new(
            data_type,
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            dims[1] * dims[2] * dims[3],
            dims[2] * dims[3],
            dims[3],
            1
        )
    }

    pub fn from_nhwc(data_type: cudnnDataType_t, dims: &[usize]) -> Result<Tensor, Error> {
        Tensor::new(
            data_type,
            dims[0],
            dims[3],
            dims[1],
            dims[2],
            dims[1] * dims[2] * dims[3],
            1,
            dims[2] * dims[3],
            dims[3]
        )
    }

    pub fn info(&self) -> Result<(cudnnDataType_t, (usize, usize, usize, usize), (usize, usize, usize, usize)), Error> {
        let (mut n, mut c, mut h, mut w) = (0, 0, 0, 0);
        let (mut n_stride, mut c_stride, mut h_stride, mut w_stride) = (0, 0, 0, 0);
        let mut data_type = cudnnDataType_t::Float;
        let success = unsafe {
            cudnnGetTensor4dDescriptor(
                self.as_ptr(),
                &mut data_type,
                &mut n,
                &mut c,
                &mut h,
                &mut w,
                &mut n_stride,
                &mut c_stride,
                &mut h_stride,
                &mut w_stride
            )
        };

        if success != 0 {
            Err(Error::CudnnError(success))
        } else {
            Ok((
                data_type,
                (n as usize, c as usize, h as usize, w as usize),
                (n_stride as usize, c_stride as usize, h_stride as usize, w_stride as usize)
            ))
        }
    }

    pub fn len(&self) -> Result<usize, Error> {
        let (_data_type, (n, c, h, w), _strides) = self.info()?;

        Ok(n * c * h * w)
    }

    pub fn size_in_bytes(&self) -> Result<usize, Error> {
        let (data_type, (n, c, h, w), _strides) = self.info()?;

        Ok(data_type.size_in_bytes() * n * c * h * w)
    }

    pub fn convert_to_ptr(&self, data: &Vec<f32>) -> Result<Ptr, Error> {
        let (data_type, _dims, _strides) = self.info()?;

        if data_type == cudnnDataType_t::Float {
            Ptr::from_vec(data, &Stream::default())
        } else if data_type == cudnnDataType_t::Half {
            let f16_data = data.iter()
                .map(|&x| f16::from(x))
                .collect();

            Ptr::from_vec(&f16_data, &Stream::default())
        } else {
            unreachable!();
        }
    }

    pub fn copy_to_host<T: Clone + Sized + Default>(&self, device_ptr: *const T, stream: &Stream) -> Result<Vec<T>, Error> {
        let mut host = vec! [T::default(); self.len()?];

        copy_nonoverlapping(
            device_ptr,
            host.as_mut_ptr(),
            host.len(),
            cudaMemcpyKind_t::DeviceToHost,
            stream
        )?;

        Ok(host)
    }
}

#[cfg(test)]
mod tests {
    // pass
}
