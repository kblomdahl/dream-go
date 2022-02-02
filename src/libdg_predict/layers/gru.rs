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

use crate::{Allocator, Err, Variable, Io};
use super::{LayerFactory, LayerImpl};

use dg_cuda::{self as cuda, cudnn};

use std::collections::HashMap;

#[derive(Default)]
pub struct GruFactory;

impl LayerFactory for GruFactory {
    fn build(
        &self,
        _handle: &cudnn::Handle,
        _variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Result<Box<dyn LayerImpl>, Err>
    {
        Ok(Box::new(Gru::new()))
    }
}

pub struct Gru;

impl Gru {
    pub fn new() -> Self {
        Self {}
    }
}

impl LayerImpl for Gru {
    fn build(
        &mut self,
        _handle: &cudnn::Handle,
        _variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        Ok(())
    }

    fn prepare(
        &mut self,
        _handle: &cudnn::Handle,
        _batch_size: i32,
        _variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        // x = inputs
        // hx = hidden_state
        // cx = unused
        // w  =
        // y  = outputs
        // hy = hidden_state
        // cy = unused
        Ok(())
    }

    fn forward(
        &self,
        _handle: &cudnn::Handle,
        _inputs: Io,
        _allocator: &mut Allocator,
        _stream: &cuda::Stream,
    ) -> Result<Io, Err>
    {
        panic!("not yet implemented");
    }
}

#[cfg(test)]
mod tests {
    // pass
}
