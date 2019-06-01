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

use graph_def::{GraphDef, LayerDef, LayerTypeDef};
use optimizers::rewriter::{GraphDefRewriter, PluckResult};

/// Removes any `Identity` layers whose inputs and outputs has the same
/// shape
pub struct RemoveRedundantIdentityLayers;

impl GraphDefRewriter for RemoveRedundantIdentityLayers {}

impl RemoveRedundantIdentityLayers {
    pub fn apply(mut g: GraphDef) -> GraphDef {

        loop {
            g = match Self::pluck_layer(g, Self::is_redundant_identity) {
                PluckResult::NotFound(g) => { return g; },
                PluckResult::Found(mut g, layer_def) => {
                    for (input_def, output_def) in layer_def.map() {
                        if output_def.id != input_def.id {
                            Self::replace_variable_with(&mut g, output_def.id, input_def.id);
                        }
                    }

                    g
                }
            };
        }
    }

    fn is_redundant_identity(_graph_def: &GraphDef, layer_def: &LayerDef) -> bool {
        layer_def.type_of == LayerTypeDef::Identity && {
            for (input_def, output_def) in layer_def.map() {
                if input_def.shape != output_def.shape {
                    return false;
                }
            }

            true
        }
    }
}
