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

use graph_def::*;
use optimizers::rewriter::{GraphDefRewriter, PluckResult};

/// Add a scale layer before each softmax to adjust the _smoothness_ of the curve.
pub struct ScaleBeforeSoftmax;

impl GraphDefRewriter for ScaleBeforeSoftmax {}

impl ScaleBeforeSoftmax {
    pub fn apply(mut g: GraphDef, alpha: f32) -> GraphDef {
        loop {
            g = match Self::pluck_layer(g, Self::is_softmax_without_scale) {
                PluckResult::NotFound(g) => { return g },
                PluckResult::Found(mut g, mut softmax_def) => {
                    for input_def in softmax_def.input.iter_mut() {
                        if !Self::is_output_of(&g, Self::is_scale, input_def) {
                            let replace_with = Self::add_scale_layer(&mut g, alpha, &input_def);

                            input_def.id = replace_with.id;
                        }
                    }

                    g.layers.push(softmax_def);
                    g
                }
            };
        }
    }

    fn add_scale_layer(g: &mut GraphDef, alpha: f32, x: &VariableDef) -> VariableDef {
        let y = Self::create_variable(g, x.data_type, x.shape.clone());

        g.layers.push(LayerDef {
            type_of: LayerTypeDef::Scale,
            input: vec! [x.clone()],
            output: vec! [y.clone()],
            arguments: Some(LayerArgumentsDef {
                kernel: None,
                bias: None,
                group_count: 1,
                alpha: Some(ConstantDef {
                    shape: vec! [1],
                    value: ConstantValueDef {
                        inner: Arc::new(vec! [alpha])
                    }
                }),
                activation: ActivationTypeDef::Linear
            })
        });

        y
    }

    fn is_softmax_without_scale(g: &GraphDef, layer_def: &LayerDef) -> bool {
        layer_def.type_of == LayerTypeDef::Softmax && {
            for input_def in &layer_def.input {
                if !Self::is_output_of(g, Self::is_scale, input_def) {
                    return false;
                }
            }

            true
        }
    }

    fn is_scale(layer_def: &LayerDef) -> bool {
        layer_def.type_of == LayerTypeDef::Scale
    }
}
