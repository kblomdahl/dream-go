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
pub struct Softmax;

impl Layer for Softmax {
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

impl PreparedLayer for Softmax {
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

        let softmax = cudnn::Softmax::new()?;

        softmax.forward(
            handle,
            inputs[0].0,
            inputs[0].1,
            outputs[0].0,
            outputs[0].1
        )
    }
}

impl Softmax {
    pub fn new(_layer_def: &LayerDef) -> Result<Softmax, cuda::Error> {
        Ok(Softmax)
    }
}

#[cfg(test)]
mod tests {
    use dg_cuda::cudnn::cudnnDataType_t;
    use graph_def::{ActivationTypeDef, LayerArgumentsDef, LayerTypeDef, VariableDef};
    use layers::tests::{assert_approx_eq, run_layer};

    use super::*;

    #[test]
    fn softmax() {
        let layer_def = LayerDef {
            type_of: LayerTypeDef::Softmax,
            input: vec! [
                VariableDef { id: 0, shape: vec! [4, 1, 1, 362] }
            ],
            output: vec! [
                VariableDef { id: 0, shape: vec! [4, 1, 1, 362] }
            ],
            arguments: Some(LayerArgumentsDef {
                kernel: None,
                bias: None,
                alpha: None,
                group_count: 0,
                activation: ActivationTypeDef::Linear
            })
        };
        let layer = Softmax::new(&layer_def)
            .expect("Could not create softmax layer");

        let (inputs, outputs) = run_layer::<f32, _>(
            &layer_def,
            &layer,
            cudnnDataType_t::Float
        );

        for i in 0..4 {
            let mut expected = vec! [0.0f32; 362];
            let mut max = ::std::f32::MIN;
            let mut sum = 0.0;

            for j in 0..362 {
                if inputs[0][362*i+j] > max {
                    max = inputs[0][362*i+j];
                }
            }

            for j in 0..362 {
                let exp_x = (inputs[0][362*i+j] - max).exp();

                expected[j] = exp_x;
                sum += exp_x;
            }

            for j in 0..362 {
                assert_approx_eq(outputs[0][362*i+j], expected[j] / sum);
            }
        }
    }
}