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
use std::sync::{Arc, Mutex};

use dg_cuda as cuda;
use dg_cuda::cudnn;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ActivationKey {
    device: cuda::Device,
    mode: cudnn::cudnnActivationMode_t,
    relu_nan_opt: cudnn::cudnnNanPropagation_t
}

lazy_static! {
    static ref ACTIVATIONS: Mutex<HashMap<ActivationKey, Arc<cudnn::Activation>>> = Mutex::new(HashMap::default());
}

pub fn get_or_create(
    mode: cudnn::cudnnActivationMode_t,
    relu_nan_opt: cudnn::cudnnNanPropagation_t,
) -> Result<Arc<cudnn::Activation>, cuda::Error>
{
    let mut activations = ACTIVATIONS.lock().unwrap();
    let key = ActivationKey {
        device: cuda::Device::current()?,
        mode,
        relu_nan_opt
    };

    if !activations.contains_key(&key) {
        activations.insert(key.clone(), Arc::new(
            cudnn::Activation::new(mode, relu_nan_opt, 1.0)?
        ));
    }

    Ok(activations[&key].clone())
}