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

use super::{Allocator, Config, ExecutionPlan};

use dg_cuda as cuda;
use dg_utils::types::f16;

use std::mem::size_of;

pub struct Io {
    pub batch_size: usize,

    pub intermediate: cuda::SmartPtr<Allocator>,
    pub hidden_states: cuda::SmartPtr<Allocator>,
    pub features: cuda::SmartPtr<Allocator>,
    pub policy: cuda::SmartPtr<Allocator>,
    pub value: cuda::SmartPtr<Allocator>
}

pub struct Output {
    pub hidden_states: Vec<f16>,
    pub policy: Vec<f16>,
    pub value: Vec<f16>
}

impl Io {
    pub fn new(batch_size: usize, config: &Config, allocator: &mut Allocator) -> Result<Self, cuda::Error> {
        Ok(Self {
            batch_size,

            intermediate: cuda::malloc(0, allocator)?,
            hidden_states: cuda::malloc(batch_size * config.embeddings_size * size_of::<f16>(), allocator)?,
            features: cuda::malloc(batch_size * config.num_features * config.image_size * config.image_size * size_of::<f16>(), allocator)?,
            policy: cuda::malloc(batch_size * 362 * size_of::<f16>(), allocator)?,
            value: cuda::malloc(batch_size * 1 * size_of::<f16>(), allocator)?
        })
    }

    pub fn current(&self) -> &cuda::Ptr {
        if !self.intermediate.is_null() {
            &self.intermediate
        } else {
            &self.features
        }
    }

    pub fn as_outputs(self, stream: &cuda::Stream) -> Result<Output, cuda::Error> {
        Ok(Output {
            hidden_states: self.hidden_states.to_vec::<f16>(stream)?,
            policy: self.policy.to_vec::<f16>(stream)?,
            value: self.value.to_vec::<f16>(stream)?,
        })
    }

    pub fn copy_features_from(mut self, features: &[f16], plan: &ExecutionPlan) -> Result<Self, cuda::Error> {
        self.features.copy_from_slice(features, &plan.features_copy_stream)?;
        plan.features_copy_finished.record(&plan.features_copy_stream)?;
        Ok(self)
    }

    pub fn copy_hidden_states_from(mut self, hidden_states: &[f16], plan: &ExecutionPlan) -> Result<Self, cuda::Error> {
        self.hidden_states.copy_from_slice(hidden_states, &plan.hidden_states_copy_stream)?;
        plan.hidden_states_copy_finished.record(&plan.hidden_states_copy_stream)?;
        Ok(self)
    }

    pub fn with_hidden_states(mut self, hidden_states: cuda::SmartPtr<Allocator>) -> Self {
        self.hidden_states = hidden_states;
        self
    }

    pub fn with_intermediate(mut self, intermediate: cuda::SmartPtr<Allocator>) -> Self {
        self.intermediate = intermediate;
        self
    }

    pub fn with_intermediate_as_hidden_states(mut self, stream: &cuda::Stream) -> Result<Self, cuda::Error> {
        debug_assert_eq!(self.intermediate.size_in_bytes(), self.hidden_states.size_in_bytes());

        self.hidden_states.copy_from_ptr(&self.intermediate, self.intermediate.size_in_bytes(), stream)?;
        Ok(self)
    }

    pub fn with_policy(mut self, policy: cuda::SmartPtr<Allocator>) -> Self {
        self.policy = policy;
        self
    }

    pub fn with_value(mut self, value: cuda::SmartPtr<Allocator>) -> Self {
        self.value = value;
        self
    }
}
