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

use std::collections::HashMap;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::Arc;

use dg_cuda as cuda;
use dg_cuda::cudnn;
use dg_go::utils::features::{FEATURE_SIZE};
use dg_utils::types::f16;
use dg_utils::per_thread::PerThread;
use crate::layers::{PolicyLayer, ResidualLayer, UpLayer, ValueLayer};
use crate::output_map::*;
use crate::tensor::Tensor;
use crate::Error;

// -------- Graph --------

pub struct Builder {
    tensors: Arc<HashMap<String, Tensor>>,
    allocator: cuda::PerDevice<PerThread<cuda::Cloneable<cuda::Sticky<cuda::Native>>>>,
}

impl Builder {
    pub fn new(tensors: HashMap<String, Tensor>) -> Builder {
        Builder {
            tensors: Arc::new(tensors),
            allocator: cuda::PerDevice::new().unwrap(),
        }
    }

    /// Returns a mutable workspace that contains everything you need to
    /// perform a forward pass through the network pre-allocated.
    ///
    /// # Arguments
    ///
    /// * `batch_size` -
    ///
    pub fn get_workspace(&self, batch_size: usize) -> Result<Workspace, Error> {
        let handle_dnn: cudnn::Handle = cudnn::Handle::new()?;
        let c_up = Rc::new(UpLayer::new(&handle_dnn, batch_size as i32, &self.tensors)?);
        let c_residual = self.get_residual_layers(&handle_dnn, batch_size)?;
        let c_value = Rc::new(ValueLayer::new(&handle_dnn, batch_size as i32, 2 + c_residual.len(), &self.tensors)?);
        let c_policy = Rc::new(PolicyLayer::new(&handle_dnn, batch_size as i32, 2 + c_residual.len(), &self.tensors)?);

        Ok(Workspace {
            batch_size: batch_size,
            allocator: self.allocator.clone(),

            handle: handle_dnn,

            tower_finished: cuda::Event::new()?,

            tower_stream: cuda::Stream::new()?,
            policy_stream: cuda::Stream::new()?,
            value_stream: cuda::Stream::new()?,

            c_up: c_up,
            c_value: c_value,
            c_policy: c_policy,
            c_residual: c_residual
        })
    }

    fn get_residual_layers(
        &self,
        handle_dnn: &cudnn::Handle,
        batch_size: usize
    ) -> Result<Vec<Rc<ResidualLayer>>, Error>
    {
        let mut c_residual = vec! [];
        let mut count = 2;

        loop {
            match ResidualLayer::new(handle_dnn, batch_size as i32, count, &self.tensors) {
                Ok(None) => { break },
                Ok(Some(layer)) => { c_residual.push(Rc::new(layer)) },
                Err(reason) => { return Err(reason) }
            }

            count += 1;
        }

        Ok(c_residual)
    }
}

pub struct Workspace {
    batch_size: usize,
    allocator: PerThread<cuda::Cloneable<cuda::Sticky<cuda::Native>>>,

    handle: cudnn::Handle,
    tower_finished: cuda::Event,
    tower_stream: cuda::Stream,
    policy_stream: cuda::Stream,
    value_stream: cuda::Stream,

    c_up: Rc<UpLayer>,
    c_value: Rc<ValueLayer>,
    c_policy: Rc<PolicyLayer>,
    c_residual: Vec<Rc<ResidualLayer>>
}

/// Returns the value and policy tensors obtained from a forward pass
/// through the neural network.
///
/// # Arguments
///
/// * `workspace` - the workspace for the current thread
/// * `features` - the input features
///
pub fn forward(workspace: &mut Workspace, features: &[f16]) -> Result<OutputMap, Error> {
    debug_assert!(features.len() % FEATURE_SIZE == 0);
    debug_assert!(features.len() / FEATURE_SIZE == workspace.batch_size);

    let allocator = &mut workspace.allocator.get().clone();

    // copy all of the input features into a temporary workspace
    let mut input = cuda::malloc(size_of::<f16>() * features.len(), allocator)?;
    input.copy_from_slice(&features, &workspace.tower_stream)?;

    // Upsample features to 128 channels
    let mut residual_1 = workspace.c_up.clone().forward(&workspace.handle, &input, allocator, &workspace.tower_stream)?;

    // residual blocks
    let num_residual = workspace.c_residual.len();

    for i in 0..num_residual {
        let residual = &workspace.c_residual[i];

        residual_1 = residual.clone().forward(&workspace.handle, residual_1, allocator, &workspace.tower_stream)?;
    }

    workspace.tower_finished.record(&workspace.tower_stream)?;
    workspace.value_stream.wait_event(&workspace.tower_finished)?;
    workspace.policy_stream.wait_event(&workspace.tower_finished)?;

    // run the value and policy head, then wait for them to finish (if
    // they are requested)
    let value = workspace.c_value.clone().forward(&workspace.handle, &residual_1, allocator, &workspace.value_stream)?;
    let policy = workspace.c_policy.clone().forward(&workspace.handle, &residual_1, allocator, &workspace.policy_stream)?;

    Ok(OutputMap::new(
        value.to_vec::<f16>(&workspace.value_stream)?,
        policy.to_vec::<f16>(&workspace.policy_stream)?
    ))
}
