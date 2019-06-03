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

#[derive(Clone, Debug)]
pub struct Scale {
    alpha: f32
}

impl Layer for Scale {
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

impl PreparedLayer for Scale {
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

        let scale = cudnn::Scale::new(self.alpha)?;

        scale.forward(
            handle,
            inputs[0].0,
            inputs[0].1,
            outputs[0].0,
            outputs[0].1
        )
    }
}

impl Scale {
    pub fn new(layer_def: &LayerDef) -> Result<Scale, cuda::Error> {
        let arguments = layer_def.arguments.as_ref().unwrap();
        let arg_alpha = arguments.alpha.as_ref().unwrap();
        let shape = &arg_alpha.shape;
        let value = &arg_alpha.value;

        assert_eq!(shape, &vec! [1]);

        Ok(Scale {
            alpha: f32::from(value.inner[0])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph_def::{LayerTypeDef, LayerArgumentsDef, VariableDef, ActivationTypeDef, ConstantDef, ConstantValueDef};
    use layers::tests::{run_layer, assert_approx_eq};
    use dg_cuda::cudnn::cudnnDataType_t;
    use std::sync::Arc;
    use dg_utils::types::f16;

    fn check_scale_by(alpha: f32) {
        let layer_def = LayerDef {
            type_of: LayerTypeDef::Scale,
            input: vec! [
                VariableDef { id: 0, shape: vec! [1, 19, 19, 64] }
            ],
            output: vec! [
                VariableDef { id: 0, shape: vec! [1, 19, 19, 64] }
            ],
            arguments: Some(LayerArgumentsDef {
                kernel: None,
                bias: None,
                alpha: Some(ConstantDef {
                    shape: vec! [ 1 ],
                    value: ConstantValueDef {
                        inner: Arc::new(vec! [ f16::from(alpha) ])
                    }
                }),
                group_count: 0,
                activation: ActivationTypeDef::Linear
            })
        };
        let layer = Scale::new(&layer_def)
            .expect("Could not create scale layer");

        let (inputs, outputs) = run_layer::<f32, _>(
            &layer_def,
            &layer,
            cudnnDataType_t::Float
        );

        for (&inp, &outp) in inputs[0].iter().zip(outputs[0].iter()) {
            assert_approx_eq(outp, inp * alpha);
        }
    }

    #[test]
    fn scale_neg() {
        check_scale_by(-1.0);
        check_scale_by(-0.5);
        check_scale_by(-1.0 / 3.0);
        check_scale_by(-0.95);
    }

    #[test]
    fn scale_pos() {
        check_scale_by(2.0);
        check_scale_by(0.95);
        check_scale_by(1.0 / 3.0);
    }
}