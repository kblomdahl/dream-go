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

use std::collections::vec_deque::VecDeque;
use std::sync::Arc;

use fnv::{FnvHashMap, FnvHashSet};

use dg_cuda as cuda;
use graph_def::{GraphDef, LayerDef, LayerTypeDef, VariableDef};
use layer::{Layer, LayerId};
use layers::*;

pub struct Graph {
    graph_def: Arc<GraphDef>,
    variables: FnvHashMap<usize, VariableDef>,
    layers: Vec<(Box<Layer>, LayerDef)>,
    pub device: cuda::Device
}

impl Graph {
    /// Returns a graph
    pub fn new(graph_def: &Arc<GraphDef>, device: cuda::Device, outputs: &[String]) -> Result<Graph, cuda::Error> {
        device.set_current()?;

        let mut out = Graph {
            graph_def: graph_def.clone(),
            variables: FnvHashMap::default(),
            layers: Vec::with_capacity(graph_def.layers.len()),
            device
        };

        // get ride of any layers that does contribute to the provided outputs
        let inputs = graph_def.input.values().collect();
        let layers = prune_layers(
            &graph_def.layers,
            &outputs.iter()
                .filter_map(|name| graph_def.output.get(name))
                .map(|variable_def| variable_def.id)
                .collect()
        );

        for (layer_def, _deps) in topologically_sorted_layers(layers, inputs) {
            for variable_def in &layer_def.input {
                out.add_variable(variable_def);
            }

            for variable_def in &layer_def.output {
                out.add_variable(variable_def);
            }

            let layer: Box<Layer> = match layer_def.type_of {
                LayerTypeDef::Activation => Box::new(Activation::new(layer_def)?),
                LayerTypeDef::Add => Box::new(OpTensor::new(layer_def)?),
                LayerTypeDef::Conv2D => Box::new(Conv2D::new(layer_def)?),
                LayerTypeDef::Dense => Box::new(Dense::new(layer_def)?),
                LayerTypeDef::GlobalAveragePooling => Box::new(GlobalAveragePooling::new(layer_def)?),
                LayerTypeDef::Identity => Box::new(Identity::new(layer_def)?),
                LayerTypeDef::Multiply => Box::new(OpTensor::new(layer_def)?),
                LayerTypeDef::Softmax => Box::new(Softmax::new(layer_def)?),
                LayerTypeDef::Scale => Box::new(Scale::new(layer_def)?),
                LayerTypeDef::Transform => Box::new(Transform::new(layer_def)?),
            };

            out.layers.push((layer, layer_def.clone()));
        }

        Ok(out)
    }

    fn add_variable(&mut self, variable_def: &VariableDef) {
        let other = self.variables.entry(variable_def.id)
            .or_insert(variable_def.clone());

        assert_eq!(other.id, variable_def.id);
        assert_eq!(other.data_type, variable_def.data_type);
        assert_eq!(other.shape, variable_def.shape);
    }

    pub fn inputs(&self) -> Vec<(&String, &VariableDef)> {
        self.graph_def.input.iter()
            .filter(|(_name, variable_def)| self.variables.contains_key(&variable_def.id))
            .collect()
    }

    pub fn outputs(&self) -> Vec<(&String, &VariableDef)> {
        self.graph_def.output.iter()
            .filter(|(_name, variable_def)| self.variables.contains_key(&variable_def.id))
            .collect()
    }

    pub fn layers(&self) -> &[(Box<Layer>, LayerDef)] {
        &self.layers
    }

    pub fn variables(&self) -> &FnvHashMap<usize, VariableDef> {
        &self.variables
    }
}

/// Returns a sub-set of the provided `layers` that contributes toward any the
/// given `outputs`.
///
/// # Arguments
///
/// * `layers` - the layers to sort
/// * `outputs` - inputs that will be provided at runtime
///
fn prune_layers<'a>(layers: &'a Vec<LayerDef>, outputs: &Vec<usize>) -> Vec<&'a LayerDef> {
    let layers_by_output: FnvHashMap<usize, Vec<usize>> = layers.iter().enumerate()
        .fold(FnvHashMap::default(), |mut acc, (i, layer_def)| {
            for output in &layer_def.output {
                acc.entry(output.id).or_insert_with(|| vec! []).push(i);
            }
            acc
        });

    // prune any layers that does not contribute to the desired `outputs`.
    let mut visited = FnvHashSet::default();
    let mut remaining = VecDeque::default();

    for output in outputs {
        if let Some(layers) = layers_by_output.get(output) {
            for layer in layers {
                remaining.push_back(layer);
            }
        }
    }

    while let Some(&n) = remaining.pop_back() {
        if visited.insert(n) {
            let layer = &layers[n];

            for input in &layer.input {
                if let Some(input_layers) = layers_by_output.get(&input.id) {
                    for input_layer in input_layers {
                        remaining.push_back(input_layer);
                    }
                }
            }
        }
    }

    visited.into_iter()
        .map(|i| &layers[i])
        .collect()
}

/// Returns a topological sorting of the given `layers`, such that they can be executed
/// in the returned order and give back the expected results.
///
/// # Arguments
///
/// * `layers` - the layers to sort
/// * `inputs` - inputs that will be provided at runtime
///
fn topologically_sorted_layers<'a>(layers: Vec<&'a LayerDef>, inputs: Vec<&VariableDef>) -> Vec<(&'a LayerDef, FnvHashSet<LayerId<'a>>)> {
    let layers_by_input: FnvHashMap<usize, Vec<usize>> = layers.iter().enumerate()
        .fold(FnvHashMap::default(), |mut acc, (i, layer_def)| {
            for input in &layer_def.input {
                acc.entry(input.id).or_insert_with(|| vec! []).push(i);
            }
            acc
        });
    let mut inputs_by_layer: FnvHashMap<usize, FnvHashSet<usize>> = layers.iter().enumerate()
        .fold(FnvHashMap::default(), |mut acc, (i, layer_def)| {
            acc.insert(i, layer_def.input.iter().map(|i| i.id).collect());
            acc
        });
    let mut layer_dependencies = FnvHashMap::default();

    // find all layers with only the provided `inputs` as incoming nodes
    let mut remaining = VecDeque::default();

    for input in inputs {
        if let Some(layers) = layers_by_input.get(&input.id) {
            for layer in layers {
                let inputs = inputs_by_layer.entry(*layer).and_modify(|inputs| {
                    inputs.remove(&input.id);
                }).or_default();

                if inputs.is_empty() {
                    layer_dependencies.insert(*layer, FnvHashSet::default());
                    remaining.push_back(*layer);
                }
            }
        }
    }

    // This is Kahn's algorithm for topological sorting, it works by visiting all nodes
    // whose predecessors has already been visited by eliminating edges progressively
    // and keeping track of the remaining edges for each node in an efficient manner.
    let mut out = vec![];

    while let Some(n) = remaining.pop_front() {
        let dependencies_n = layer_dependencies[&n].clone();
        out.push(n);

        for output in &layers[n].output {
            if let Some(layers) = layers_by_input.get(&output.id) {
                for &layer in layers.iter() {
                    let inputs = inputs_by_layer.entry(layer).and_modify(|inputs| {
                        let dependencies = layer_dependencies.entry(layer)
                            .or_insert_with(|| FnvHashSet::default());

                        dependencies.insert(n);
                        for &dependency in &dependencies_n {
                            dependencies.insert(dependency);
                        }

                        inputs.remove(&output.id);
                    }).or_default();

                    if inputs.is_empty() {
                        remaining.push_back(layer);
                    }
                }
            }
        }
    }

    // check that all edges in the graph has been traversed at this
    // point, i.e. that there were no cycles.
    for (_i, inputs) in inputs_by_layer {
        assert!(inputs.is_empty(), "expect no incoming edges to remain -- {:?}", inputs);
    }

    out.into_iter()
        .map(|i| {
            let dependencies = layer_dependencies[&i].iter()
                .map(|&j| LayerId { id: j, layer_def: &layers[j] })
                .collect();

            (layers[i], dependencies)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    // pass
}