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

use libc::c_void;

use ::graph_def::LayerDef;
use ::layer::{Layer, PreparedLayer};
use dg_cuda as cuda;
use dg_cuda::cudnn;
use std::sync::Arc;
use factories::pooling_factory;

#[derive(Clone, Debug)]
pub struct GlobalAveragePooling {
    pooling: Arc<cudnn::Pooling>
}

impl Layer for GlobalAveragePooling {
    fn prepare(
        &self,
        _handle: &cudnn::Handle,
        _inputs: &[&cudnn::Tensor],
        _outputs: &[&cudnn::Tensor]
    ) -> Result<Box<PreparedLayer>, cuda::Error>
    {
        Ok(Box::new(self.clone()))
    }
}

impl PreparedLayer for GlobalAveragePooling {
    fn size_in_bytes(&self) -> usize {
        0
    }

    fn forward(
        &self,
        handle: &cudnn::Handle,
        inputs: &[(&cudnn::Tensor, *const c_void)],
        outputs: &[(&cudnn::Tensor, *mut c_void)],
        _workspace_ptr: *mut c_void
    ) -> Result<(), cuda::Error>
    {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        self.pooling.forward(
            handle,
            inputs[0].0,
            inputs[0].1,
            outputs[0].0,
            outputs[0].1
        )
    }
}

impl GlobalAveragePooling {
    pub fn new(layer_def: &LayerDef) -> Result<GlobalAveragePooling, cuda::Error> {
        let shape = &layer_def.input[0].shape;

        Ok(GlobalAveragePooling {
            pooling: pooling_factory::get_or_create(
                cudnn::cudnnPoolingMode_t::AvgCountExcludePadding,
                cudnn::cudnnNanPropagation_t::NotPropagateNaN,
                (shape[1] as usize, shape[2] as usize),
                (0, 0),
                (1, 1)
            )?
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph_def::{LayerTypeDef, LayerArgumentsDef, VariableDef, ActivationTypeDef, DataTypeDef};
    use layers::tests::{run_layer, assert_approx_eq};
    use dg_utils::types::f16;

    fn global_avg_pool<T>(data_type: DataTypeDef)
        where T: Copy + Default + From<f32>,
              f32: From<T>
    {
        let layer_def = LayerDef {
            type_of: LayerTypeDef::GlobalAveragePooling,
            input: vec! [
                VariableDef { id: 0, shape: vec! [1, 19, 19, 16], data_type }
            ],
            output: vec! [
                VariableDef { id: 0, shape: vec! [1, 1, 1, 16], data_type }
            ],
            arguments: Some(LayerArgumentsDef {
                kernel: None,
                bias: None,
                alpha: None,
                group_count: 0,
                activation: ActivationTypeDef::Linear
            })
        };
        let layer = GlobalAveragePooling::new(&layer_def)
            .expect("Could not create global average pool layer");

        let (inputs, outputs) = run_layer::<T, T, _>(
            &layer_def,
            &layer
        );

        for i in 0..16 {
            let mut expected_output: f32 = 0.0f32;

            for j in 0..361 {
                expected_output += f32::from(inputs[0][i + 16*j]);
            }

            assert_approx_eq::<f32>(f32::from(outputs[0][i]), expected_output / 361.0);
        }
    }

    #[test]
    fn global_avg_pool_float() {
        global_avg_pool::<f32>(DataTypeDef::Float)
    }

    #[test]
    fn global_avg_pool_half() {
        global_avg_pool::<f16 >(DataTypeDef::Half)
    }
}