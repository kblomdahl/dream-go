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

use std::sync::Arc;

use libc::c_void;

use ::graph_def::{ActivationTypeDef, LayerDef};
use ::layer::Layer;
use dg_cuda as cuda;
use dg_cuda::cudnn;
use factories::activation_factory;
use layer::PreparedLayer;

#[derive(Clone, Debug)]
pub struct Activation {
    activation: Arc<cudnn::Activation>
}

impl Layer for Activation {
    fn prepare(
        &self,
        _handle: &cudnn::Handle,
        _inputs: &[&cudnn::Tensor],
        _outputs: &[&cudnn::Tensor]
    ) -> Result<Box<PreparedLayer>, cuda::Error>
    {
        Ok(Box::new(self.clone()))
    }
}

impl PreparedLayer for Activation {
    fn size_in_bytes(&self) -> usize {
        0
    }

    fn forward(
        &self,
        handle: &cudnn::Handle,
        inputs: &[(&cudnn::Tensor, *const c_void)],
        outputs: &[(&cudnn::Tensor, *mut c_void)],
        _workspace_ptr: *mut c_void
    ) -> Result<(), cuda::Error>
    {
        debug_assert_eq!(inputs.len(), 1);
        debug_assert_eq!(outputs.len(), 1);

        self.activation.forward(
            handle,
            inputs[0].0,
            inputs[0].1,
            outputs[0].0,
            outputs[0].1
        )
    }
}

impl Activation {
    pub fn new(layer_def: &LayerDef) -> Result<Activation, cuda::Error> {
        let mode: String = match layer_def.arguments {
            None => "linear".into(),
            Some(ref args) => {
                match args.activation {
                    ActivationTypeDef::ReLU => "relu".into(),
                    ActivationTypeDef::Sigmoid => "sigmoid".into(),
                    ActivationTypeDef::Linear => "linear".into(),
                }
            }
        };

        Self::with_mode(&mode)
    }

    pub fn with_mode(mode: &str) -> Result<Activation, cuda::Error> {
        let mode = match mode {
            "relu" => cudnn::cudnnActivationMode_t::ReLU,
            "sigmoid" => cudnn::cudnnActivationMode_t::Sigmoid,
            "tanh" => cudnn::cudnnActivationMode_t::Tanh,
            _ => cudnn::cudnnActivationMode_t::Identity,
        };

        Ok(Activation {
            activation: activation_factory::get_or_create(
                mode,
                cudnn::cudnnNanPropagation_t::NotPropagateNaN,
            )?
        })
    }
}