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

use libc::{c_void, size_t};

#[link(name = "cudnn_cnn_infer")]
extern {
    fn cudnnConvolutionBiasActivationForward(
        handle: cudnnHandle_t,
        alpha_1: *const c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const c_void,
        w_desc: cudnnFilterDescriptor_t,
        w: *const c_void,
        conv_desc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        workspace: *const c_void,
        workspace_size_in_bytes: size_t,
        alpha_2: *const c_void,
        z_desc: cudnnTensorDescriptor_t,
        z: *const c_void,
        offset_desc: cudnnTensorDescriptor_t,
        offset: *const c_void,
        activation_desc: cudnnActivationDescriptor_t,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut c_void
    ) -> cudnnStatus_t;
}

pub struct ConvolutionBiasActivation {
    x: TensorDescriptor,
    y: TensorDescriptor,
    z: TensorDescriptor,
    w: FilterDescriptor,
    conv: ConvolutionDescriptor,
    offset: TensorDescriptor,
    activation: ActivationDescriptor,
    fwd_algo_perf: ConvolutionFwdAlgoPerf,
    alpha: [f32; 2]
}

impl ConvolutionBiasActivation {
    pub fn new(
        handle: &Handle,
        alpha_1: f32,
        x: TensorDescriptor,
        w: FilterDescriptor,
        conv: ConvolutionDescriptor,
        alpha_2: f32,
        offset: TensorDescriptor,
        activation: ActivationDescriptor,
        y: TensorDescriptor,
        z: TensorDescriptor,
    ) -> Result<Self, Status>
    {
        let fwd_algo_perf = ConvolutionFwdAlgoPerf::new(handle, &x, &w, &conv, &y)?;
        let alpha = [alpha_1, alpha_2];

        Ok(Self {
            x,
            y,
            z,
            w,
            conv,
            offset,
            activation,
            fwd_algo_perf,
            alpha
        })
    }

    pub fn raw_forward(
        handle: &Handle,
        alpha_1: *const c_void,
        x_desc: &TensorDescriptor,
        x: *const c_void,
        w_desc: &FilterDescriptor,
        w: *const c_void,
        conv_desc: &ConvolutionDescriptor,
        algo: ConvolutionFwdAlgo,
        workspace: *const c_void,
        workspace_size_in_bytes: usize,
        alpha_2: *const c_void,
        z_desc: &TensorDescriptor,
        z: *const c_void,
        offset_desc: &TensorDescriptor,
        offset: *const c_void,
        activation_desc: &ActivationDescriptor,
        y_desc: &TensorDescriptor,
        y: *mut c_void
    ) -> Result<(), Status>
    {
        let status =
            unsafe {
                cudnnConvolutionBiasActivationForward(
                    **handle,
                    alpha_1,
                    **x_desc, x,
                    **w_desc, w,
                    **conv_desc, algo,
                    workspace, workspace_size_in_bytes,
                    alpha_2,
                    **z_desc, z,
                    **offset_desc, offset,
                    **activation_desc,
                    **y_desc, y
                )
            };

        status.into_result(())
    }

    pub fn forward(
        &self,
        handle: &Handle,
        x_data: *const c_void,
        w_data: *const c_void,
        workspace: *const c_void,
        workspace_size_in_bytes: usize,
        z_data: *const c_void,
        offset_data: *const c_void,
        y_data: *mut c_void
    ) -> Result<(), Status>
    {
        Self::raw_forward(
            handle,
            &self.alpha[0] as *const _ as *const c_void,
            &self.x, x_data,
            &self.w, w_data,
            &self.conv, self.fwd_algo_perf.algo(),
            workspace, workspace_size_in_bytes,
            &self.alpha[1] as *const _ as *const c_void,
            &self.z, z_data,
            &self.offset, offset_data,
            &self.activation,
            &self.y, y_data
        )
    }

    pub fn input(&self) -> &TensorDescriptor {
        &self.x
    }

    pub fn offset(&self) -> &TensorDescriptor {
        &self.offset
    }

    pub fn output(&self) -> &TensorDescriptor {
        &self.y
    }

    pub fn fwd_algo_perf(&self) -> &ConvolutionFwdAlgoPerf {
        &self.fwd_algo_perf
    }
}

#[cfg(test)]
mod tests {
    // pass
}
