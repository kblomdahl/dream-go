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

use graph_def::{GraphDef, LayerDef, VariableDef};
use fnv::FnvHashSet;

pub enum PluckResult {
    Found(GraphDef, LayerDef),
    NotFound(GraphDef)
}

pub trait GraphDefRewriter {
    fn pluck_layer<F>(mut g: GraphDef, pred: F) -> PluckResult
        where F: Fn(&GraphDef, &LayerDef) -> bool
    {
        let index = g.layers.iter()
            .position(|layer_def| pred(&g, layer_def));

        if let Some(index) = index {
            let layer_def = g.layers.swap_remove(index);

            PluckResult::Found(g, layer_def)
        } else {
            PluckResult::NotFound(g)
        }
    }

    fn create_variable(g: &GraphDef, shape: Vec<isize>) -> VariableDef {
        let mut unique_ids = FnvHashSet::default();

        for layer_def in &g.layers {
            for variable_def in layer_def.variables() {
                unique_ids.insert(variable_def.id);
            }
        }

        VariableDef {
            id: (0..=unique_ids.len())
                .rev()
                .filter(|id| !unique_ids.contains(id))
                .next().unwrap(),
            shape
        }
    }

    fn replace_variable_with(g: &mut GraphDef, to_replace_id: usize, replace_with_id: usize) {
        assert_ne!(to_replace_id, replace_with_id);

        for output_def in g.output.values_mut() {
            if output_def.id == to_replace_id {
                output_def.id = replace_with_id;
            }
        }

        for input_def in g.input.values() {
            assert_ne!(input_def.id, to_replace_id);
        }

        for layer_def in g.layers.iter_mut() {
            for input_def in layer_def.input.iter_mut() {
                if input_def.id == to_replace_id {
                    input_def.id = replace_with_id;
                }
            }

            for output_def in layer_def.output.iter() {
                assert_ne!(output_def.id, to_replace_id);
            }
        }
    }

    fn is_output_of<F>(g: &GraphDef, pred: F, x: &VariableDef) -> bool
        where F: Fn(&LayerDef) -> bool
    {
        g.layers.iter()
            .any(|layer_def| {
                layer_def.output.iter().any(|output_def| output_def.id == x.id) &&
                    pred(layer_def)
            })
    }
}
