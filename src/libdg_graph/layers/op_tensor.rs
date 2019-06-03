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

use std::sync::Arc;

use libc::c_void;

use ::graph_def::{LayerDef, LayerTypeDef};
use ::layer::{Layer, PreparedLayer};
use dg_cuda as cuda;
use dg_cuda::cudnn;
use factories::op_tensor_factory;

#[derive(Clone, Debug)]
pub struct OpTensor {
    op_tensor: Arc<cudnn::OpTensor>
}

impl Layer for OpTensor {
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

impl PreparedLayer for OpTensor {
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
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);

        self.op_tensor.forward(
            &handle,
            inputs[0].0,
            inputs[0].1,
            inputs[1].0,
            inputs[1].1,
            outputs[0].0,
            outputs[0].1
        )
    }
}

impl OpTensor {
    pub fn new(layer_def: &LayerDef) -> Result<OpTensor, cuda::Error> {
        let op = match layer_def.type_of {
            LayerTypeDef::Add => cudnn::cudnnOpTensorOp_t::Add,
            LayerTypeDef::Multiply => cudnn::cudnnOpTensorOp_t::Mul,
            _ => { panic!(); }
        };

        Ok(OpTensor {
            op_tensor: op_tensor_factory::get_or_create(
                op,
                cudnn::cudnnDataType_t::Float,
                cudnn::cudnnNanPropagation_t::NotPropagateNaN
            )?
        })
    }
}

#[cfg(test)]
mod tests {
    use dg_cuda::cudnn::cudnnDataType_t;
    use graph_def::{ActivationTypeDef, LayerArgumentsDef, LayerTypeDef, VariableDef};
    use layers::tests::{assert_approx_eq, run_layer};

    use super::*;

    fn execute_op(layer_type_def: LayerTypeDef, input_2: Vec<isize>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let layer_def = LayerDef {
            type_of: layer_type_def,
            input: vec! [
                VariableDef { id: 0, shape: vec! [2, 19, 19, 16] },
                VariableDef { id: 0, shape: input_2 },
            ],
            output: vec! [
                VariableDef { id: 0, shape: vec! [2, 19, 19, 16] }
            ],
            arguments: Some(LayerArgumentsDef {
                kernel: None,
                bias: None,
                alpha: None,
                group_count: 0,
                activation: ActivationTypeDef::Linear
            })
        };
        let layer = OpTensor::new(&layer_def)
            .expect("Could not create op tensor layer");

        run_layer::<f32, _>(
            &layer_def,
            &layer,
            cudnnDataType_t::Float
        )
    }

    #[test]
    fn add() {
        let (inputs, outputs) = execute_op(
            LayerTypeDef::Add,
            vec! [2, 19, 19, 16]
        );

        for i in 0..outputs[0].len() {
            assert_approx_eq(outputs[0][i], inputs[0][i] + inputs[1][i]);
        }
    }

    #[test]
    fn add_broadcast() {
        let (inputs, outputs) = execute_op(
            LayerTypeDef::Add,
            vec! [1, 1, 1, 16]
        );

        for i in 0..outputs[0].len() {
            assert_approx_eq(outputs[0][i], inputs[0][i] + inputs[1][i % 16]);
        }
    }

    #[test]
    fn mul() {
        let (inputs, outputs) = execute_op(
            LayerTypeDef::Multiply,
            vec! [2, 19, 19, 16]
        );

        for i in 0..outputs[0].len() {
            assert_approx_eq(outputs[0][i], inputs[0][i] * inputs[1][i]);
        }
    }

    #[test]
    fn mul_broadcast() {
        let (inputs, outputs) = execute_op(
            LayerTypeDef::Multiply,
            vec! [1, 1, 1, 16]
        );

        for i in 0..outputs[0].len() {
            assert_approx_eq(outputs[0][i], inputs[0][i] * inputs[1][i % 16]);
        }
    }
}