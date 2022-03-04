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

use crate::{Allocator, Err};

use dg_cuda::{self as cuda, cudnn, cublas_lt, Event, Stream};

pub struct ExecutionPlan {
    pub allocator: Allocator,
    pub light_handle: cublas_lt::Handle,
    pub handle: cudnn::Handle,

    pub hidden_states_copy_finished: Event,
    pub hidden_states_copy_stream: Stream,

    pub features_copy_finished: Event,
    pub features_copy_stream: Stream,

    pub stream: Stream,
}

impl ExecutionPlan {
    pub fn new(allocator: &cuda::Concurrent<cuda::Sticky<cuda::Native>>) -> Result<Self, Err> {
        let light_handle = cublas_lt::Handle::new()?;
        let handle = cudnn::Handle::new()?;
        let stream = Stream::new()?;
        handle.set_stream(&stream)?;

        Ok(Self {
            allocator: cuda::Cloneable::new(cuda::Sticky::new(allocator.clone())),
            light_handle,
            handle,

            hidden_states_copy_finished: Event::new()?,
            hidden_states_copy_stream: Stream::new()?,

            features_copy_finished: Event::new()?,
            features_copy_stream: Stream::new()?,

            stream,
        })
    }
}
