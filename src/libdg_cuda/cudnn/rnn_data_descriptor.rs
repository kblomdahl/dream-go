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

use std::ptr::{self, null, null_mut};
use std::ops::Deref;
use libc::{c_void, c_int};

#[allow(non_camel_case_types)]
pub type cudnnRNNDataDescriptor_t = *const c_void;

#[link(name = "cudnn_adv_infer")]
extern {
    fn cudnnCreateRNNDataDescriptor(rnn_data_desc: *mut cudnnRNNDataDescriptor_t) -> cudnnStatus_t;
    fn cudnnDestroyRNNDataDescriptor(rnn_data_desc: cudnnRNNDataDescriptor_t) -> cudnnStatus_t;
    fn cudnnSetRNNDataDescriptor(
        rnn_data_desc: cudnnRNNDataDescriptor_t,
        data_type: cudnnDataType_t,
        layout: cudnnRNNDataLayout_t,
        max_seq_length: c_int,
        batch_size: c_int,
        vector_size: c_int,
        seqLengthArray: *const c_int,
        paddingFill: *const c_void
    ) -> cudnnStatus_t;

    fn cudnnGetRNNDataDescriptor(
        rnn_data_desc: cudnnRNNDataDescriptor_t,
        data_type: *mut cudnnDataType_t,
        layout: *mut cudnnRNNDataLayout_t,
        max_seq_length: *mut c_int,
        batch_size: *mut c_int,
        vector_size: *mut c_int,
        array_length_requested: c_int,
        seq_length_array: *mut c_int,
        padding_fill: *mut c_void
    ) -> cudnnStatus_t;
}

struct GetRnnDataDescriptor {
    data_type: cudnnDataType_t,
    layout: cudnnRNNDataLayout_t,
    max_seq_length: c_int,
    batch_size: c_int,
    vector_size: c_int,
    padding_fill: [u8; 8]
}

impl GetRnnDataDescriptor {
    fn new(rnn_data_desc: cudnnRNNDataDescriptor_t) -> Result<Self, Status> {
        let mut out = Self {
            data_type: cudnnDataType_t::Float,
            layout: cudnnRNNDataLayout_t::SeqMajorPacked,
            max_seq_length: 0,
            batch_size: 0,
            vector_size: 0,
            padding_fill: [0; 8]
        };
        let status  = unsafe {
            cudnnGetRNNDataDescriptor(
                rnn_data_desc,
                &mut out.data_type,
                &mut out.layout,
                &mut out.max_seq_length,
                &mut out.batch_size,
                &mut out.vector_size,
                0,
                null_mut(),
                out.padding_fill.as_mut_ptr() as *mut _
            )
        };

        status.into_result(out)
    }
}

pub struct RnnDataDescriptor {
    rnn_data_desc: cudnnRNNDataDescriptor_t
}

unsafe impl Send for RnnDataDescriptor {}

impl Drop for RnnDataDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyRNNDataDescriptor(self.rnn_data_desc) };
    }
}

impl Deref for RnnDataDescriptor {
    type Target = cudnnRNNDataDescriptor_t;

    fn deref(&self) -> &Self::Target {
        &self.rnn_data_desc
    }
}

impl RnnDataDescriptor {
    pub fn empty() -> Result<Self, Status> {
        let mut out = Self {
            rnn_data_desc: ptr::null_mut()
        };
        let status = unsafe { cudnnCreateRNNDataDescriptor(&mut out.rnn_data_desc) };

        status.into_result(out)
    }

    pub fn new(
        data_type: cudnnDataType_t,
        layout: cudnnRNNDataLayout_t,
        max_seq_length: c_int,
        batch_size: c_int,
        vector_size: c_int,
        seq_length_array: &[i32]
    ) -> Result<Self, Status>
    {
        debug_assert_eq!(batch_size as usize, seq_length_array.len());

        Self::empty().and_then(|out| {
            let status = unsafe {
                cudnnSetRNNDataDescriptor(
                    out.rnn_data_desc,
                    data_type,
                    layout,
                    max_seq_length,
                    batch_size,
                    vector_size,
                    seq_length_array.as_ptr(),
                    null()
                )
            };

            status.into_result(out)
        })
    }

    pub fn batch_size(&self) -> Result<i32, Status> {
        GetRnnDataDescriptor::new(self.rnn_data_desc)
            .map(|out| out.batch_size)
    }

    pub fn size_in_bytes(&self) -> Result<usize, Status> {
        GetRnnDataDescriptor::new(self.rnn_data_desc)
            .map(|out| (out.max_seq_length * out.batch_size * out.vector_size) as usize * out.data_type.size_in_bytes())
    }
}
