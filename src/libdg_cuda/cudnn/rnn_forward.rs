// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use std::ptr::{null, null_mut};
use libc::{c_void, size_t};

#[link(name = "cudnn_adv_infer")]
extern {
    fn cudnnRNNForward(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        fwd_mode: cudnnForwardMode_t,
        dev_seq_lengths: *const i32,
        x_desc: cudnnRNNDataDescriptor_t,
        x: *const c_void,
        y_desc: cudnnRNNDataDescriptor_t,
        y: *mut c_void,
        h_desc: cudnnTensorDescriptor_t,
        hx: *const c_void,
        hy: *mut c_void,
        c_desc: cudnnTensorDescriptor_t,
        cx: *const c_void,
        cy: *mut c_void,
        weight_space_size: size_t,
        weight_space: *const c_void,
        workspace_size: size_t,
        workspace: *mut c_void,
        reserve_space_size: size_t,
        reserve_space: *mut c_void
    ) -> cudnnStatus_t;

    fn cudnnGetRNNTempSpaceSizes(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        f_mode: cudnnForwardMode_t,
        x_desc: cudnnRNNDataDescriptor_t,
        workSpaceSize: *mut size_t,
        reserveSpaceSize: *mut size_t
    ) -> cudnnStatus_t;
}

pub struct RnnForward {
    rnn_desc: RnnDescriptor,
    x_desc: RnnDataDescriptor,
    y_desc: RnnDataDescriptor,
    h_desc: TensorDescriptor
}

impl RnnForward {
    pub fn new(
        rnn_desc: RnnDescriptor,
        x_desc: RnnDataDescriptor,
        y_desc: RnnDataDescriptor,
        h_desc: TensorDescriptor
    ) -> Result<Self, Status>
    {
        Ok(Self {
            rnn_desc,
            x_desc,
            y_desc,
            h_desc
        })
    }

    pub fn rnn_desc(&self) -> &cudnnRNNDescriptor_t {
        &self.rnn_desc
    }

    pub fn x(&self) -> &RnnDataDescriptor {
        &self.x_desc
    }

    pub fn y(&self) -> &RnnDataDescriptor {
        &self.y_desc
    }

    pub fn h(&self) -> &TensorDescriptor {
        &self.h_desc
    }

    pub fn workspace_size_in_bytes(&self, handle: &Handle) -> Result<usize, Status> {
        let mut workspace_size_in_bytes = 0;
        let status = unsafe {
            cudnnGetRNNTempSpaceSizes(
                **handle,
                *self.rnn_desc,
                cudnnForwardMode_t::Inference,
                *self.x_desc,
                &mut workspace_size_in_bytes,
                null_mut()
            )
        };

        status.into_result(workspace_size_in_bytes)
    }

    pub fn forward(
        &self,
        handle: &Handle,
        dev_seq_lengths: *const i32,
        x_data: *const c_void,
        y_data: *mut c_void,
        hx_data: *const c_void,
        hy_data: *mut c_void,
        weight_space_size: size_t,
        weight_space: *const c_void,
        workspace_size: size_t,
        workspace: *mut c_void
    ) -> Result<(), Status>
    {
        let status = unsafe {
            cudnnRNNForward(
                **handle,
                *self.rnn_desc,
                RnnForwardMode::Inference,
                dev_seq_lengths,
                *self.x_desc,
                x_data,
                *self.y_desc,
                y_data,
                *self.h_desc,
                hx_data,
                hy_data,
                null(),
                null(),
                null_mut(),
                weight_space_size,
                weight_space,
                workspace_size,
                workspace,
                0,
                null_mut()
            )
        };

        status.into_result(())
    }
}
