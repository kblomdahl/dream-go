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
pub struct OpTensorKey {
    device: cuda::Device,
    op: cudnn::cudnnOpTensorOp_t,
    comp_type: cudnn::cudnnDataType_t,
    nan_opt: cudnn::cudnnNanPropagation_t
}

lazy_static! {
    static ref OP_TENSORS: Mutex<HashMap<OpTensorKey, Arc<cudnn::OpTensor>>> = Mutex::new(HashMap::default());
}

pub fn get_or_create(
    op: cudnn::cudnnOpTensorOp_t,
    comp_type: cudnn::cudnnDataType_t,
    nan_opt: cudnn::cudnnNanPropagation_t
) -> Result<Arc<cudnn::OpTensor>, cuda::Error>
{
    let mut op_tensors = OP_TENSORS.lock().unwrap();
    let key = OpTensorKey {
        device: cuda::Device::current()?,
        op,
        comp_type,
        nan_opt
    };

    if !op_tensors.contains_key(&key) {
        op_tensors.insert(key.clone(), Arc::new(
            cudnn::OpTensor::new(op, comp_type, nan_opt)?
        ));
    }

    Ok(op_tensors[&key].clone())
}