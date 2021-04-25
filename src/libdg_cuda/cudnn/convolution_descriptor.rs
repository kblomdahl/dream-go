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
use libc::{c_int, c_void};

#[allow(non_camel_case_types)]
pub type cudnnConvolutionDescriptor_t = *const c_void;

#[link(name = "cudnn_cnn_infer")]
extern {
    fn cudnnCreateConvolutionDescriptor(conv_desc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
    fn cudnnDestroyConvolutionDescriptor(conv_desc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
    fn cudnnSetConvolution2dDescriptor(
        conv_desc: cudnnConvolutionDescriptor_t,
        pad_h: c_int,
        pad_w: c_int,
        u: c_int,
        v: c_int,
        dilation_h: c_int,
        dilation_w: c_int,
        mode: cudnnConvolutionMode_t,
        compute_type: cudnnDataType_t
    ) -> cudnnStatus_t;
    fn cudnnGetConvolution2dDescriptor(
        conv_desc: cudnnConvolutionDescriptor_t,
        pad_h: *mut c_int,
        pad_w: *mut c_int,
        u: *mut c_int,
        v: *mut c_int,
        dilation_h: *mut c_int,
        dilation_w: *mut c_int,
        mode: *mut cudnnConvolutionMode_t,
        compute_type: *mut cudnnDataType_t
    ) -> cudnnStatus_t;
    fn cudnnSetConvolutionMathType(
        conv_desc: cudnnConvolutionDescriptor_t,
        math_type: cudnnMathType_t
    ) -> cudnnStatus_t;
    fn cudnnGetConvolutionMathType(
        conv_desc: cudnnConvolutionDescriptor_t,
        math_type: *mut cudnnMathType_t
    ) -> cudnnStatus_t;
    fn cudnnSetConvolutionReorderType(
        conv_desc: cudnnConvolutionDescriptor_t,
        reorder_type: cudnnReorderType_t
    ) -> cudnnStatus_t;
    fn cudnnGetConvolutionReorderType(
        conv_desc: cudnnConvolutionDescriptor_t,
        reorder_type: *mut cudnnReorderType_t
    ) -> cudnnStatus_t;
}

struct GetConvolution2dDescriptor {
    pad_h: c_int,
    pad_w: c_int,
    u: c_int,
    v: c_int,
    dilation_h: c_int,
    dilation_w: c_int,
    mode: cudnnConvolutionMode_t,
    compute_type: cudnnDataType_t
}

impl GetConvolution2dDescriptor {
    fn new(conv_desc: cudnnConvolutionDescriptor_t) -> Result<Self, Status> {
        let mut out = GetConvolution2dDescriptor {
            pad_h: 0,
            pad_w: 0,
            u: 0,
            v: 0,
            dilation_h: 0,
            dilation_w: 0,
            mode: cudnnConvolutionMode_t::Convolution,
            compute_type: cudnnDataType_t::Float
        };
        let status =
            unsafe {
                cudnnGetConvolution2dDescriptor(
                    conv_desc,
                    &mut out.pad_h,
                    &mut out.pad_w,
                    &mut out.u,
                    &mut out.v,
                    &mut out.dilation_h,
                    &mut out.dilation_w,
                    &mut out.mode,
                    &mut out.compute_type
                )
            };

        status.into_result(out)
    }
}

pub struct ConvolutionDescriptor {
    conv_desc: cudnnConvolutionDescriptor_t
}

impl Deref for ConvolutionDescriptor {
    type Target = cudnnConvolutionDescriptor_t;

    fn deref(&self) -> &Self::Target {
        &self.conv_desc
    }
}

impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyConvolutionDescriptor(self.conv_desc) };
    }
}

impl ConvolutionDescriptor {
    pub fn empty() -> Result<Self, Status> {
        let mut out = Self { conv_desc: ptr::null_mut() };
        let status = unsafe { cudnnCreateConvolutionDescriptor(&mut out.conv_desc) };

        status.into_result(out)
    }

    pub fn new(
        pad: &[i32],
        stride: &[i32],
        dilation: &[i32],
        mode: ConvolutionMode,
        compute_type: DataType
    ) -> Result<Self, Status>
    {
        debug_assert_eq!(pad.len(), 2);
        debug_assert_eq!(stride.len(), 2);
        debug_assert_eq!(dilation.len(), 2);

        Self::empty().and_then(|out| {
            let status =
                unsafe {
                    cudnnSetConvolution2dDescriptor(
                        out.conv_desc,
                        pad[0],
                        pad[1],
                        stride[0],
                        stride[1],
                        dilation[0],
                        dilation[1],
                        mode,
                        compute_type
                    )
                };

            out.set_math_type()?;
            status.into_result(out)
        })
    }

    fn set_math_type(&self) -> Result<MathType, Status> {
        if supports_tensor_cores()? {
            let status =
                unsafe {
                    cudnnSetConvolutionMathType(
                        self.conv_desc,
                        MathType::TensorOpMath
                    )
                };

            status.into_result(MathType::TensorOpMath)
        } else {
            Ok(MathType::DefaultMath)
        }
    }

    pub fn math_type(&self) -> Result<MathType, Status> {
        let mut out = MathType::DefaultMath;
        let status = unsafe { cudnnGetConvolutionMathType(self.conv_desc, &mut out) };

        status.into_result(out)
    }

    pub fn set_reorder_type(&self, reorder_type: ReorderType) -> Result<(), Status> {
        let status =
            unsafe {
                cudnnSetConvolutionReorderType(
                    self.conv_desc,
                    reorder_type
                )
            };

        status.into_result(())
    }

    pub fn reorder_type(&self) -> Result<ReorderType, Status> {
        let mut out = ReorderType::DefaultReorder;
        let status = unsafe { cudnnGetConvolutionReorderType(self.conv_desc, &mut out) };

        status.into_result(out)
    }

    pub fn pad(&self) -> Result<Vec<i32>, Status> {
        GetConvolution2dDescriptor::new(self.conv_desc).map(|out| vec! [out.pad_h, out.pad_w])
    }

    pub fn stride(&self) -> Result<Vec<i32>, Status> {
        GetConvolution2dDescriptor::new(self.conv_desc).map(|out| vec! [out.u, out.v])
    }

    pub fn dilation(&self) -> Result<Vec<i32>, Status> {
        GetConvolution2dDescriptor::new(self.conv_desc).map(|out| vec! [out.dilation_h, out.dilation_w])
    }

    pub fn mode(&self) -> Result<ConvolutionMode, Status> {
        GetConvolution2dDescriptor::new(self.conv_desc).map(|out| out.mode)
    }

    pub fn compute_type(&self) -> Result<DataType, Status> {
        GetConvolution2dDescriptor::new(self.conv_desc).map(|out| out.compute_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create() {
        let conv_desc = ConvolutionDescriptor::new(
            &[0, 0],
            &[1, 1],
            &[1, 1],
            ConvolutionMode::CrossCorrelation,
            DataType::Float
        );

        assert!(conv_desc.is_ok());
    }

    #[test]
    fn get_conv_descriptor() {
        let conv_desc = ConvolutionDescriptor::new(
            &[1, 1],
            &[2, 2],
            &[3, 3],
            ConvolutionMode::CrossCorrelation,
            DataType::Half
        ).unwrap();

        assert_eq!(conv_desc.math_type(), Ok(if supports_tensor_cores().unwrap() { MathType::TensorOpMath } else { MathType::DefaultMath }));
        assert_eq!(conv_desc.pad(), Ok(vec! [1, 1]));
        assert_eq!(conv_desc.stride(), Ok(vec! [2, 2]));
        assert_eq!(conv_desc.dilation(), Ok(vec! [3, 3]));
        assert_eq!(conv_desc.mode(), Ok(ConvolutionMode::CrossCorrelation));
        assert_eq!(conv_desc.compute_type(), Ok(DataType::Half));
    }
}
