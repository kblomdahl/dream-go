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
use libc::c_void;

#[allow(non_camel_case_types)]
pub type cudnnReduceTensorDescriptor_t = *const c_void;

#[link(name = "cudnn_ops_infer")]
extern {
    fn cudnnCreateReduceTensorDescriptor(reduce_tensor_desc: *mut cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;
    fn cudnnDestroyReduceTensorDescriptor(reduce_tensor_desc: cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;
    fn cudnnSetReduceTensorDescriptor(
        reduce_tensor_desc: cudnnReduceTensorDescriptor_t,
        reduce_tensor_op: cudnnReduceTensorOp_t,
        reduce_tensor_comp_type: cudnnDataType_t,
        reduce_tensor_nan_opt: cudnnNanPropagation_t,
        reduce_tensor_indices: cudnnReduceTensorIndices_t,
        reduce_tensor_indices_type: cudnnIndicesType_t
    ) -> cudnnStatus_t;

    fn cudnnGetReduceTensorDescriptor(
        reduce_tensor_desc: cudnnReduceTensorDescriptor_t,
        reduce_tensor_op: *mut cudnnReduceTensorOp_t,
        reduce_comp_type: *mut cudnnDataType_t,
        reduce_tensor_nan_opt: *mut cudnnNanPropagation_t,
        reduce_tensor_indices: *mut cudnnReduceTensorIndices_t,
        reduce_tensor_indices_type: *mut cudnnIndicesType_t
    ) -> cudnnStatus_t;
}

struct GetReduceTensorDescriptor {
    op: cudnnReduceTensorOp_t,
    comp_type: cudnnDataType_t,
    nan_opt: cudnnNanPropagation_t,
    indices: cudnnReduceTensorIndices_t,
    indices_type: cudnnIndicesType_t
}

impl GetReduceTensorDescriptor {
    fn new(reduce_tensor_desc: cudnnReduceTensorDescriptor_t) -> Result<Self, Status> {
        let mut out = Self {
            op: cudnnReduceTensorOp_t::Add,
            comp_type: cudnnDataType_t::Float,
            nan_opt: cudnnNanPropagation_t::NotPropagateNaN,
            indices: cudnnReduceTensorIndices_t::NoIndices,
            indices_type: cudnnIndicesType_t::_32
        };
        let status = unsafe {
            cudnnGetReduceTensorDescriptor(
                reduce_tensor_desc,
                &mut out.op,
                &mut out.comp_type,
                &mut out.nan_opt,
                &mut out.indices,
                &mut out.indices_type
            )
        };

        status.into_result(out)
    }
}

pub struct ReduceTensorDescriptor {
    reduce_tensor_desc: cudnnReduceTensorDescriptor_t
}

unsafe impl Send for ReduceTensorDescriptor {}

impl Drop for ReduceTensorDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyReduceTensorDescriptor(self.reduce_tensor_desc) };
    }
}

impl Deref for ReduceTensorDescriptor {
    type Target = cudnnReduceTensorDescriptor_t;

    fn deref(&self) -> &Self::Target {
        &self.reduce_tensor_desc
    }
}

impl ReduceTensorDescriptor {
    fn empty() -> Result<Self, Status> {
        let mut out = Self { reduce_tensor_desc: ptr::null_mut() };
        let status = unsafe { cudnnCreateReduceTensorDescriptor(&mut out.reduce_tensor_desc) };

        status.into_result(out)
    }

    /// Returns a reduction descriptor created by `cudnnSetReduceTensorDescriptor`
    /// with the given `op`, `comp_type`, `nan_opt`, `indices`, and
    /// `indices_type`.
    ///
    /// # Arguments
    ///
    /// * `op` -
    /// * `comp_type` -
    /// * `nan_opt` -
    /// * `indices` -
    /// * `indices_type` -
    ///
    pub fn new(
        op: ReduceTensorOp,
        comp_type: DataType,
        nan_opt: NanPropagation,
        indices: ReduceTensorIndices,
        indices_type: IndicesType
    ) -> Result<Self, Status> {
        let out = Self::empty()?;
        let status = unsafe {
            cudnnSetReduceTensorDescriptor(
                out.reduce_tensor_desc,
                op,
                comp_type,
                nan_opt,
                indices,
                indices_type
            )
        };

        status.into_result(out)
    }

    pub fn compute_type(&self) -> Result<DataType, Status> {
        GetReduceTensorDescriptor::new(self.reduce_tensor_desc).map(|out| out.comp_type)
    }

    pub fn nan_opt(&self) -> Result<NanPropagation, Status> {
        GetReduceTensorDescriptor::new(self.reduce_tensor_desc).map(|out| out.nan_opt)
    }

    pub fn indices(&self) -> Result<ReduceTensorIndices, Status> {
        GetReduceTensorDescriptor::new(self.reduce_tensor_desc).map(|out| out.indices)
    }

    pub fn indices_type(&self) -> Result<IndicesType, Status> {
        GetReduceTensorDescriptor::new(self.reduce_tensor_desc).map(|out| out.indices_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_reduce_tensor_desc() {
        let reduce_tensor_desc = ReduceTensorDescriptor::new(
            ReduceTensorOp::Avg,
            DataType::Float,
            NanPropagation::NotPropagateNaN,
            ReduceTensorIndices::NoIndices,
            IndicesType::_32
        );

        assert!(reduce_tensor_desc.is_ok());

        let reduce_tensor_desc = reduce_tensor_desc.unwrap();

        assert_eq!(reduce_tensor_desc.compute_type(), Ok(DataType::Float));
        assert_eq!(reduce_tensor_desc.nan_opt(), Ok(NanPropagation::NotPropagateNaN));
        assert_eq!(reduce_tensor_desc.indices(), Ok(ReduceTensorIndices::NoIndices));
        assert_eq!(reduce_tensor_desc.indices_type(), Ok(IndicesType::_32));
    }
}
