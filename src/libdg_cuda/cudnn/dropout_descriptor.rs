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

use std::ptr::null;
use std::ops::Deref;
use libc::{c_void, size_t, c_ulonglong};

#[allow(non_camel_case_types)]
pub type cudnnDropoutDescriptor_t = *const c_void;

#[link(name = "cudnn_adv_infer")]
extern {
    fn cudnnCreateDropoutDescriptor(rnn_desc: *mut cudnnDropoutDescriptor_t) -> cudnnStatus_t;
    fn cudnnDestroyDropoutDescriptor(rnn_desc: cudnnDropoutDescriptor_t) -> cudnnStatus_t;
    fn cudnnSetDropoutDescriptor(
        dropout_desc: cudnnDropoutDescriptor_t,
        handle: cudnnHandle_t,
        dropout: f32,
        states: *const c_void,
        state_size_in_bytes: size_t,
        seed: c_ulonglong
    ) -> cudnnStatus_t;
}

/// `cudnnDropoutDescriptor_t` is a pointer to an opaque structure holding the
/// description of a dropout operation. cudnnCreateDropoutDescriptor() is used
/// to create one instance, cudnnSetDropoutDescriptor() is used to initialize
/// this instance, cudnnDestroyDropoutDescriptor() is used to destroy this
/// instance, cudnnGetDropoutDescriptor() is used to query fields of a previously
/// initialized instance, cudnnRestoreDropoutDescriptor() is used to restore an
/// instance to a previously saved off state.
pub struct DropoutDescriptor {
    dropout_desc: cudnnDropoutDescriptor_t
}

unsafe impl Send for DropoutDescriptor {}

impl Drop for DropoutDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyDropoutDescriptor(self.dropout_desc) };
    }
}

impl DropoutDescriptor {
    pub fn empty() -> Result<Self, Status> {
        let mut out = Self { dropout_desc: null() };
        let status = unsafe { cudnnCreateDropoutDescriptor(&mut out.dropout_desc) };

        status.into_result(out)
    }

    pub fn new(
        handle: &Handle,
        dropout: f32
    ) -> Result<Self, Status>
    {
        Self::empty().and_then(|out| {
            let status = unsafe {
                cudnnSetDropoutDescriptor(
                    out.dropout_desc,
                    **handle,
                    dropout,
                    null(),
                    0,
                    0
                )
            };

            status.into_result(out)
        })
    }
}

impl Deref for DropoutDescriptor {
    type Target = cudnnDropoutDescriptor_t;

    fn deref(&self) -> &Self::Target {
        &self.dropout_desc
    }
}
