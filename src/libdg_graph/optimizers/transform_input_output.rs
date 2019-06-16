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

use optimizers::rewriter::GraphDefRewriter;
use graph_def::{GraphDef, VariableDef, DataTypeDef, LayerDef, LayerTypeDef};
use fnv::FnvHashSet;

/// The input and output should always be half precision, so ensure that they
/// are by adding `Transform` layers at the appropriate places.
pub struct TransformInputOutput;

impl GraphDefRewriter for TransformInputOutput {}

impl TransformInputOutput {
    pub fn apply(mut g: GraphDef) -> GraphDef {
        let inputs = g.input.values()
            .map(|var_def| var_def.id)
            .collect::<Vec<_>>();

        for var_id in inputs.into_iter() {
            g = Self::ensure_input_is_half(g, var_id);
        }

        let outputs = g.output.values()
            .map(|var_def| var_def.id)
            .collect::<Vec<_>>();

        for var_id in outputs.into_iter() {
            g = Self::ensure_output_is_float(g, var_id);
        }

        g
    }

    fn ensure_input_is_half(mut g: GraphDef, var_id: usize) -> GraphDef {
        let data_types = Self::get_data_types(&g, var_id);
        assert_eq!(data_types.len(), 1);

        if data_types.contains(&DataTypeDef::Float) {
            let input_var = Self::get_variable_def(&g, var_id);
            let other_var = Self::create_variable(
                &g,
                DataTypeDef::Half,
                input_var.shape.clone()
            );

            g.layers.push(LayerDef {
                type_of: LayerTypeDef::Transform,
                input: vec! [other_var.clone()],
                output: vec! [input_var],
                arguments: None
            });

            Self::replace_input_output_with(
                &mut g,
                var_id,
                other_var.id
            );
        }

        g
    }

    fn ensure_output_is_float(mut g: GraphDef, var_id: usize) -> GraphDef {
        let data_types = Self::get_data_types(&g, var_id);
        assert_eq!(data_types.len(), 1);

        if data_types.contains(&DataTypeDef::Half) {
            let input_var = Self::get_variable_def(&g, var_id);
            let other_var = Self::create_variable(
                &g,
                DataTypeDef::Float,
                input_var.shape.clone()
            );

            g.layers.push(LayerDef {
                type_of: LayerTypeDef::Transform,
                input: vec! [input_var],
                output: vec! [other_var.clone()],
                arguments: None
            });

            Self::replace_input_output_with(
                &mut g,
                var_id,
                other_var.id
            );
        }

        g
    }

    fn get_data_types(g: &GraphDef, var_id: usize) -> FnvHashSet<DataTypeDef> {
        let mut data_types = FnvHashSet::default();

        for layer_def in &g.layers {
            for var_def in layer_def.variables() {
                if var_def.id == var_id {
                    data_types.insert(var_def.data_type);
                }
            }
        }

        data_types
    }

    fn get_variable_def(g: &GraphDef, var_id: usize) -> VariableDef {
        for layer_def in &g.layers {
            for var_def in layer_def.variables() {
                if var_def.id == var_id {
                    return var_def.clone();
                }
            }
        }

        unreachable!();
    }

    fn replace_input_output_with(g: &mut GraphDef, to_replace_id: usize, replace_with_id: usize) {
        assert_ne!(to_replace_id, replace_with_id);

        for output_def in g.output.values_mut() {
            if output_def.id == to_replace_id {
                output_def.id = replace_with_id;
            }
        }

        for input_def in g.input.values_mut() {
            if input_def.id == to_replace_id {
                input_def.id = replace_with_id;
            }
        }
    }
}
