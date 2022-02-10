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

use crate::{cudnn::*, Ptr};

use std::convert::TryInto;
use std::ptr::{self, null};
use std::ops::Deref;
use libc::{c_void, size_t};

const CUDNN_RNN_PADDED_IO_ENABLED: u32 = 1;

#[allow(non_camel_case_types)]
pub type cudnnRNNDescriptor_t = *const c_void;

#[link(name = "cudnn_adv_infer")]
extern {
    fn cudnnCreateRNNDescriptor(rnn_desc: *mut cudnnRNNDescriptor_t) -> cudnnStatus_t;
    fn cudnnDestroyRNNDescriptor(rnn_desc: cudnnRNNDescriptor_t) -> cudnnStatus_t;
    fn cudnnSetRNNDescriptor_v8(
        rnn_desc: cudnnRNNDescriptor_t,
        algo: cudnnRNNAlgo_t,
        cell_mode: cudnnRNNMode_t,
        bias_mode: cudnnRNNBiasMode_t,
        dir_mode: cudnnDirectionMode_t,
        input_mode: cudnnRNNInputMode_t,
        data_type: cudnnDataType_t,
        math_prec: cudnnDataType_t,
        math_type: cudnnMathType_t,
        input_size: i32,
        hidden_size: i32,
        proj_size: i32,
        num_layers: i32,
        dropout_desc: cudnnDropoutDescriptor_t,
        aux_flags: u32
    ) -> cudnnStatus_t;

    fn cudnnGetRNNDescriptor_v8(
        rnn_desc: cudnnRNNDescriptor_t,
        algo: *mut cudnnRNNAlgo_t,
        cell_mode: *mut cudnnRNNMode_t,
        bias_mode: *mut cudnnRNNBiasMode_t,
        dir_mode: *mut cudnnDirectionMode_t,
        input_mode: *mut cudnnRNNInputMode_t,
        data_type: *mut cudnnDataType_t,
        math_prec: *mut cudnnDataType_t,
        math_type: *mut cudnnMathType_t,
        input_size: *mut i32,
        hidden_size: *mut i32,
        proj_size: *mut i32,
        num_layers: *mut i32,
        dropout_desc: *mut cudnnDropoutDescriptor_t,
        aux_flags: *mut u32
    ) -> cudnnStatus_t;

    fn cudnnGetRNNWeightSpaceSize(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        weightSpaceSize: *mut size_t
    ) -> cudnnStatus_t;

    fn cudnnGetRNNWeightParams(
        handle: cudnnHandle_t,
        rnn_desc: cudnnRNNDescriptor_t,
        pseudo_layer: i32,
        weight_space_size: size_t,
        weight_space: *const c_void,
        linLayerID: i32,
        m_desc: cudnnTensorDescriptor_t,
        m_addr: *mut *const c_void,
        b_desc: cudnnTensorDescriptor_t,
        b_addr: *mut *const c_void
    ) -> cudnnStatus_t;
}

struct GetRnnDescriptor {
    algo: cudnnRNNAlgo_t,
    cell_mode: cudnnRNNMode_t,
    bias_mode: cudnnRNNBiasMode_t,
    dir_mode: cudnnDirectionMode_t,
    input_mode: cudnnRNNInputMode_t,
    data_type: cudnnDataType_t,
    math_prec: cudnnDataType_t,
    math_type: cudnnMathType_t,
    input_size: i32,
    hidden_size: i32,
    proj_size: i32,
    num_layers: i32,
    dropout_desc: cudnnDropoutDescriptor_t,
    aux_flags: u32
}

impl GetRnnDescriptor {
    fn new(rnn_desc: cudnnRNNDescriptor_t) -> Result<Self, Status> {
        let mut out = Self {
            algo: cudnnRNNAlgo_t::Standard,
            cell_mode: cudnnRNNMode_t::RnnRelu,
            bias_mode: cudnnRNNBiasMode_t::NoBias,
            dir_mode: cudnnDirectionMode_t::UniDirectional,
            input_mode: cudnnRNNInputMode_t::LinearInput,
            data_type: cudnnDataType_t::Float,
            math_prec: cudnnDataType_t::Float,
            math_type: cudnnMathType_t::DefaultMath,
            input_size: -1,
            hidden_size: -1,
            proj_size: -1,
            num_layers: -1,
            dropout_desc: null(),
            aux_flags: 0
        };
        let status = unsafe {
            cudnnGetRNNDescriptor_v8(
                rnn_desc,
                &mut out.algo,
                &mut out.cell_mode,
                &mut out.bias_mode,
                &mut out.dir_mode,
                &mut out.input_mode,
                &mut out.data_type,
                &mut out.math_prec,
                &mut out.math_type,
                &mut out.input_size,
                &mut out.hidden_size,
                &mut out.proj_size,
                &mut out.num_layers,
                &mut out.dropout_desc,
                &mut out.aux_flags
            )
        };

        status.into_result(out)
    }
}

/// `cudnnRNNDescriptor_t` is a pointer to an opaque structure holding the
/// description of an RNN operation. cudnnCreateRNNDescriptor() is used to
/// create one instance.
pub struct RnnDescriptor {
    rnn_desc: cudnnRNNDescriptor_t,
    dropout_desc: DropoutDescriptor
}

unsafe impl Send for RnnDescriptor {}

impl Drop for RnnDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyRNNDescriptor(self.rnn_desc) };
    }
}

impl Deref for RnnDescriptor {
    type Target = cudnnRNNDescriptor_t;

    fn deref(&self) -> &Self::Target {
        &self.rnn_desc
    }
}

impl RnnDescriptor {
    pub fn empty(handle: &Handle) -> Result<Self, Status> {
        let mut out = Self {
            rnn_desc: ptr::null_mut(),
            dropout_desc: DropoutDescriptor::new(handle, 0.0)?
        };
        let status = unsafe { cudnnCreateRNNDescriptor(&mut out.rnn_desc) };

        status.into_result(out)
    }

    pub fn new(
        handle: &Handle,
        algo: cudnnRNNAlgo_t,
        cell_mode: cudnnRNNMode_t,
        bias_mode: cudnnRNNBiasMode_t,
        input_mode: cudnnRNNInputMode_t,
        data_type: cudnnDataType_t,
        math_prec: cudnnDataType_t,
        math_type: cudnnMathType_t,
        input_size: i32,
        hidden_size: i32,
        proj_size: i32,
        num_layers: i32,
    ) -> Result<Self, Status>
    {
        Self::empty(handle).and_then(|out| {
            let status =
                unsafe {
                    cudnnSetRNNDescriptor_v8(
                        out.rnn_desc,
                        algo,
                        cell_mode,
                        bias_mode,
                        cudnnDirectionMode_t::UniDirectional,
                        input_mode,
                        data_type,
                        math_prec,
                        math_type,
                        input_size,
                        hidden_size,
                        proj_size,
                        num_layers,
                        *out.dropout_desc,
                        CUDNN_RNN_PADDED_IO_ENABLED
                    )
                };

            status.into_result(out)
        })
    }

    pub fn algo(&self) -> Result<RnnAlgo, Status> {
        GetRnnDescriptor::new(self.rnn_desc).map(|out| out.algo)
    }

    pub fn input_size(&self) -> Result<i32, Status> {
        GetRnnDescriptor::new(self.rnn_desc).map(|out| out.input_size)
    }

    pub fn hidden_size(&self) -> Result<i32, Status> {
        GetRnnDescriptor::new(self.rnn_desc).map(|out| out.hidden_size)
    }

    pub fn proj_size(&self) -> Result<i32, Status> {
        GetRnnDescriptor::new(self.rnn_desc).map(|out| out.proj_size)
    }

    pub fn num_layers(&self) -> Result<i32, Status> {
        GetRnnDescriptor::new(self.rnn_desc).map(|out| out.num_layers)
    }

    pub fn weight_space_size(&self, handle: &Handle) -> Result<usize, Status> {
        let mut weight_space_size = 0;
        let status = unsafe {
            cudnnGetRNNWeightSpaceSize(
                **handle,
                self.rnn_desc,
                &mut weight_space_size
            )
        };

        status.into_result(weight_space_size)
    }

    pub fn weight_param_for_layer_id(&self, handle: &Handle, pseudo_layer: i32, layer_id: i32, weight_space: &Ptr) -> Result<[(TensorDescriptor, usize); 2], Status> {
        let bias_desc = TensorDescriptor::empty()?;
        let mut bias_addr = null();
        let kernel_desc = TensorDescriptor::empty()?;
        let mut kernel_addr = null();
        let weight_space_addr = weight_space.as_ptr();
        let status = unsafe {
            cudnnGetRNNWeightParams(
                **handle,
                self.rnn_desc,
                pseudo_layer,
                weight_space.size_in_bytes(),
                weight_space.as_ptr(),
                layer_id,
                *kernel_desc,
                &mut kernel_addr,
                *bias_desc,
                &mut bias_addr
            )
        };

        status.into_result([
            (
                kernel_desc,
                unsafe { kernel_addr.offset_from(weight_space_addr) }.try_into().unwrap_or(0)
            ),
            (
                bias_desc,
                unsafe { bias_addr.offset_from(weight_space_addr) }.try_into().unwrap_or(0)
            )
        ])
    }

    /// Returns the offset for the `kernel` and `bias` for each of the linear layers in the GRU
    /// layer. The layers correspond to:
    ///
    /// 1. Reset Gate
    /// 2. Update
    /// 3. New Hidden State
    /// 4. Reset Gate (recurrent)
    /// 5. Update (recurrent)
    /// 6. New Hidden State (recurrent)
    ///
    /// # Arguments
    ///
    /// * `handle` - cuDNN handle
    ///
    pub fn weight_params(&self, handle: &Handle, weight_space: &Ptr) -> Result<[[(TensorDescriptor, usize); 2]; 6], Status> {
        Ok([
            self.weight_param_for_layer_id(handle, 0, 0, weight_space)?,
            self.weight_param_for_layer_id(handle, 0, 1, weight_space)?,
            self.weight_param_for_layer_id(handle, 0, 2, weight_space)?,
            self.weight_param_for_layer_id(handle, 0, 3, weight_space)?,
            self.weight_param_for_layer_id(handle, 0, 4, weight_space)?,
            self.weight_param_for_layer_id(handle, 0, 5, weight_space)?,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_rnn_desc() {
        let handle = Handle::new().expect("could not create handle");
        let rnn_desc = RnnDescriptor::new(
            &handle,
            cudnnRNNAlgo_t::PersistStatic,
            cudnnRNNMode_t::Gru,
            cudnnRNNBiasMode_t::DoubleBias,
            cudnnRNNInputMode_t::LinearInput,
            cudnnDataType_t::Half,
            cudnnDataType_t::Float,
            cudnnMathType_t::TensorOpMath,
            722,
            722,
            722,
            2
        );

        assert!(rnn_desc.is_ok());

        let rnn = rnn_desc.unwrap();
        assert_eq!(rnn.input_size(), Ok(722));
        assert_eq!(rnn.hidden_size(), Ok(722));
        assert_eq!(rnn.proj_size(), Ok(722));
        assert_eq!(rnn.num_layers(), Ok(2));
    }

    #[test]
    fn check_weight_space_params() {
        let handle = Handle::new().expect("could not create handle");
        let rnn_desc = RnnDescriptor::new(
            &handle,
            cudnnRNNAlgo_t::PersistStatic,
            cudnnRNNMode_t::Gru,
            cudnnRNNBiasMode_t::DoubleBias,
            cudnnRNNInputMode_t::LinearInput,
            cudnnDataType_t::Half,
            cudnnDataType_t::Float,
            cudnnMathType_t::TensorOpMath,
            722,
            722,
            722,
            2
        ).expect("could not create rnn descriptor");

        let weight_space_size = rnn_desc.weight_space_size(&handle)
            .expect("could not get weight space size");
        let weight_space = Ptr::new(weight_space_size)
            .expect("could not allocate weight_space");

        assert!(weight_space.size_in_bytes() > 0);

        let params = rnn_desc.weight_params(&handle, &weight_space)
            .expect("could not get weight space params");

        for i in 0..params.len() {
            assert_eq!(params[i][0].0.shape().expect("could not get shape of kernel"), [1, 722, 722, 0]);
            assert_eq!(params[i][1].0.shape().expect("could not get shape of offset"), [1, 722, 1, 0]);
        }
    }
}
