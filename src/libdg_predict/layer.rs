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

use dg_cuda::{self as cuda, cudnn, cublas_lt};

use crate::layers::LayerFactory;

use super::layers::LayerImpl;
use super::{Allocator, Err, Variable, Io};

use std::collections::HashMap;
use std::sync::Mutex;

enum LayerInner {
    Factory { factory: Box<dyn LayerFactory> },
    Impl { layer_impl: Box<dyn LayerImpl> }
}

pub struct Layer {
    name: String,
    variables: HashMap<String, Variable>,
    inner: Mutex<LayerInner>
}

impl Layer {
    pub fn new<L>(name: &str, layer_factory: L, variables: HashMap<String, Variable>) -> Self
        where L: LayerFactory + 'static
    {
        Self {
            name: name.to_string(),
            variables,
            inner: Mutex::new(LayerInner::Factory { factory: Box::new(layer_factory) }),
        }
    }

    pub fn build(
        &mut self,
        light_handle: &cublas_lt::Handle,
        handle: &cudnn::Handle,
        stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        let mut inner = self.inner.lock().expect("poison error");

        match &mut *inner {
            LayerInner::Factory { factory } => {
                let mut layer_impl = factory.build(handle, &self.variables, stream)?;
                match layer_impl.build(light_handle, handle, &self.variables, stream) {
                    Err(Err::MissingVariable(path)) => Err(Err::MissingVariable(format!("{}/{}", self.name, path))),
                    other => other
                }?;
                *inner = LayerInner::Impl { layer_impl };
            },
            LayerInner::Impl { layer_impl } => {
                match layer_impl.build(light_handle, handle, &self.variables, stream) {
                    Err(Err::MissingVariable(path)) => Err(Err::MissingVariable(format!("{}/{}", self.name, path))),
                    other => other
                }?;
            },
        };

        Ok(())
    }

    pub fn forward(
        &self,
        light_handle: &cublas_lt::Handle,
        handle: &cudnn::Handle,
        inputs: Io,
        allocator: &mut Allocator,
        stream: &cuda::Stream,
    ) -> Result<Io, Err>
    {
        let mut inner = self.inner.lock().expect("poison error");

        match &mut *inner {
            LayerInner::Impl { layer_impl } => {
                layer_impl.prepare(light_handle, handle, inputs.batch_size as i32, &self.variables, stream)?;
                layer_impl.forward(light_handle, handle, inputs, allocator, stream)
            },
            _ => { unreachable!() }
        }
    }
}
