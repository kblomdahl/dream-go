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
pub struct PoolingKey {
    device: cuda::Device,
    mode: cudnn::cudnnPoolingMode_t,
    nan_opt: cudnn::cudnnNanPropagation_t,
    window: (usize, usize),
    padding: (usize, usize),
    stride: (usize, usize)
}

lazy_static! {
    static ref POOLINGS: Mutex<HashMap<PoolingKey, Arc<cudnn::Pooling>>> = Mutex::new(HashMap::default());
}

pub fn get_or_create(
    mode: cudnn::cudnnPoolingMode_t,
    nan_opt: cudnn::cudnnNanPropagation_t,
    window: (usize, usize),
    padding: (usize, usize),
    stride: (usize, usize)
) -> Result<Arc<cudnn::Pooling>, cuda::Error>
{
    let mut poolings = POOLINGS.lock().unwrap();
    let key = PoolingKey {
        device: cuda::Device::current()?,
        mode,
        nan_opt,
        window,
        padding,
        stride
    };

    if !poolings.contains_key(&key) {
        poolings.insert(key.clone(), Arc::new(
            cudnn::Pooling::new(
                mode,
                nan_opt,
                window,
                padding,
                stride
            )?
        ));
    }

    Ok(poolings[&key].clone())
}
