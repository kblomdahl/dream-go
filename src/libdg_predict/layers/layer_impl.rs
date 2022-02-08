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

use dg_cuda::{self as cuda, cudnn, cublas_lt};
use std::collections::HashMap;

pub trait LayerFactory : Send {
    fn build(
        &self,
        handle: &cudnn::Handle,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Result<Box<dyn LayerImpl>, Err>;
}

pub trait LayerImpl : Send {
    fn build(
        &mut self,
        light_handle: &cublas_lt::Handle,
        handle: &cudnn::Handle,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Result<(), Err>;

    fn prepare(
        &mut self,
        light_handle: &cublas_lt::Handle,
        handle: &cudnn::Handle,
        batch_size: i32,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Result<(), Err>;

    fn forward(
        &self,
        light_handle: &cublas_lt::Handle,
        handle: &cudnn::Handle,
        inputs: Io,
        allocator: &mut Allocator,
        stream: &cuda::Stream,
    ) -> Result<Io, Err>;
}
