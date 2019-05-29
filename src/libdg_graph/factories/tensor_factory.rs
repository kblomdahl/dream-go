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
pub struct TensorKey {
    device: cuda::Device,
    data_type: cudnn::cudnnDataType_t,
    tensor_format: cudnn::cudnnTensorFormat_t,
    dims: Vec<usize>,
}

lazy_static! {
    static ref TENSORS: Mutex<HashMap<TensorKey, Arc<cudnn::Tensor>>> = Mutex::new(HashMap::default());
}

pub fn get_or_create(
    data_type: cudnn::cudnnDataType_t,
    tensor_format: cudnn::cudnnTensorFormat_t,
    dims: &[usize]
) -> Result<Arc<cudnn::Tensor>, cuda::Error>
{
    let mut tensors = TENSORS.lock().unwrap();
    let key = TensorKey {
        device: cuda::Device::current()?,
        data_type,
        tensor_format,
        dims: dims.to_vec(),
    };

    if !tensors.contains_key(&key) {
        tensors.insert(key.clone(), Arc::new(
            match tensor_format {
                cudnn::cudnnTensorFormat_t::NHWC => cudnn::Tensor::from_nhwc(data_type, dims)?,
                cudnn::cudnnTensorFormat_t::NCHW => cudnn::Tensor::from_nchw(data_type, dims)?
            }
        ));
    }

    Ok(tensors[&key].clone())
}