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
pub struct ConvolutionKey {
    device: cuda::Device,
    filter_dims: Vec<usize>,
    group_count: usize
}

lazy_static! {
    static ref CONVOLUTIONS: Mutex<HashMap<ConvolutionKey, Arc<cudnn::Convolution>>> = Mutex::new(HashMap::default());
}

pub fn get_or_create(
    filter: &cudnn::Filter,
    group_count: usize
) -> Result<Arc<cudnn::Convolution>, cuda::Error>
{
    let (k, c, h, w) = filter.dims()?;
    let mut convolutions = CONVOLUTIONS.lock().unwrap();
    let key = ConvolutionKey {
        device: cuda::Device::current()?,
        filter_dims: vec! [k, c, h, w],
        group_count
    };

    if !convolutions.contains_key(&key) {
        convolutions.insert(key.clone(), Arc::new(
            cudnn::Convolution::new(filter, group_count)?
        ));
    }

    Ok(convolutions[&key].clone())
}