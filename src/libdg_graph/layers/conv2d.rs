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

use libc::c_void;

use ::graph_def::{ActivationTypeDef, LayerDef};
use ::layer::{Layer, PreparedLayer};
use dg_cuda as cuda;
use dg_cuda::cudnn;
use factories::{tensor_factory, filter_factory, activation_factory, convolution_factory};
use std::sync::Arc;

#[derive(Debug)]
pub struct Conv2D {
    bias: Arc<cudnn::Tensor>,
    bias_data: Arc<cuda::Ptr>,
    filter: Arc<cudnn::Filter>,
    filter_data: Arc<cuda::Ptr>,
    convolution: Arc<cudnn::Convolution>,
    activation: Arc<cudnn::Activation>,
}

struct PreparedConv2D {
    bias: Arc<cudnn::Tensor>,
    bias_data: Arc<cuda::Ptr>,
    filter: Arc<cudnn::Filter>,
    filter_data: Arc<cuda::Ptr>,
    convolution: Arc<cudnn::Convolution>,
    activation: Arc<cudnn::Activation>,
    fwd_algo: cudnn::cudnnConvolutionFwdAlgoPerf_t
}

impl Layer for Conv2D {
    fn prepare(
        &self,
        handle: &cudnn::Handle,
        inputs: &[&cudnn::Tensor],
        outputs: &[&cudnn::Tensor]
    ) -> Result<Box<PreparedLayer>, cuda::Error>
    {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let fwd_algo = self.convolution.compile(
            handle,
            inputs[0],
            &self.filter,
            outputs[0],
            &self.activation
        )?;

        Ok(Box::new(PreparedConv2D {
            bias: self.bias.clone(),
            bias_data: self.bias_data.clone(),
            filter: self.filter.clone(),
            filter_data: self.filter_data.clone(),
            convolution: self.convolution.clone(),
            activation: self.activation.clone(),
            fwd_algo
        }))
    }
}

impl PreparedLayer for PreparedConv2D {
    fn size_in_bytes(&self) -> usize {
        self.fwd_algo.memory
    }

    fn forward(
        &self,
        handle: &cudnn::Handle,
        inputs: &[(&cudnn::Tensor, *const c_void)],
        outputs: &[(&cudnn::Tensor, *mut c_void)],
        workspace_ptr: *mut c_void
    ) -> Result<(), cuda::Error>
    {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        self.convolution.forward(
            handle,
            inputs[0].0, inputs[0].1,
            &self.filter, self.filter_data.as_ptr(),
            &self.fwd_algo, workspace_ptr,
            &self.bias, self.bias_data.as_ptr(),
            &self.activation,
            outputs[0].0, outputs[0].1
        )
    }
}

impl Conv2D {
    pub fn new(layer_def: &LayerDef) -> Result<Conv2D, cuda::Error> {
        assert!(layer_def.arguments.is_some());

        let arguments = layer_def.arguments.as_ref().unwrap();
        let mode = match arguments.activation {
            ActivationTypeDef::ReLU => cudnn::cudnnActivationMode_t::ReLU,
            ActivationTypeDef::Sigmoid => cudnn::cudnnActivationMode_t::Sigmoid,
            ActivationTypeDef::Linear => cudnn::cudnnActivationMode_t::Identity,
        };

        Self::with_activation(
            layer_def,
            activation_factory::get_or_create(mode, cudnn::cudnnNanPropagation_t::NotPropagateNaN)?
        )
    }

    fn with_activation(layer_def: &LayerDef, activation: Arc<cudnn::Activation>) -> Result<Conv2D, cuda::Error> {
        let arguments = layer_def.arguments.as_ref().unwrap();
        let arg_group_count = arguments.group_count;
        let arg_kernel = arguments.kernel.as_ref().unwrap();
        let arg_bias = arguments.bias.as_ref().unwrap();

        let filter_dims: Vec<usize> = arg_kernel.shape.iter()
            .map(|&x| x as usize)
            .collect();
        let is_1x1 = filter_dims[1] == 1 && filter_dims[2] == 1;
        let filter = filter_factory::get_or_create(
            cudnn::cudnnDataType_t::Half,
            if is_1x1 { cudnn::cudnnTensorFormat_t::NCHW } else { cudnn::cudnnTensorFormat_t::NHWC },
            &vec! [filter_dims[0], filter_dims[3], filter_dims[1], filter_dims[2]]
        )?;
        let convolution = convolution_factory::get_or_create(&filter, arg_group_count)?;
        let bias_size = arg_bias.shape[0] as usize;
        let bias = tensor_factory::get_or_create(
            cudnn::cudnnDataType_t::Half,
            cudnn::cudnnTensorFormat_t::NCHW,
            &[1, bias_size, 1, 1]
        )?;

        Ok(Conv2D {
            activation,
            bias,
            bias_data: Arc::new(cuda::Ptr::from_vec(&arg_bias.value, &cuda::Stream::default())?),
            filter,
            filter_data: Arc::new(cuda::Ptr::from_vec(&arg_kernel.value, &cuda::Stream::default())?),
            convolution
        })
    }
}