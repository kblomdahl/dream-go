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
use std::sync::Arc;

use fnv::{FnvHashMap, FnvHashSet};
use libc::c_void;

use dg_cuda as cuda;
use dg_cuda::{cudnn, StreamGraph};
use dg_utils::types::f16;
use factories::tensor_factory;
use graph::Graph;
use graph_def::{LayerDef, VariableDef};
use layer::PreparedLayer;

pub struct Session {
    graph: Arc<Graph>,
    variables: FnvHashMap<usize, Arc<cuda::Ptr>>,
    workspace: cuda::Ptr,

    handle: cudnn::Handle,
    stream: cuda::Stream,
    by_batch_size: Vec<PreparedSession>
}

pub struct PreparedSession {
    tensors: FnvHashMap<usize, Arc<cudnn::Tensor>>,
    layers: Vec<(Box<PreparedLayer>, LayerDef)>,

    graph: Option<cuda::Graph>,
    graph_exec: Option<cuda::GraphExec>,
}

impl PreparedSession {
    fn new(
        handle: &cudnn::Handle,
        graph: &Arc<Graph>,
        batch_size: usize
    ) -> Result<PreparedSession, cuda::Error>
    {
        let num_layers = graph.layers().len();
        let mut tensor_by_shape = FnvHashMap::default();
        let mut prepared_session = PreparedSession {
            tensors: FnvHashMap::default(),
            layers: Vec::with_capacity(num_layers),
            graph: None,
            graph_exec: None
        };

        for variable_def in graph.variables().values() {
            let shape: Vec<usize> = variable_def.shape.iter()
                .map(|&x| if x < 0 { batch_size } else { x as usize })
                .collect();
            let shape = match shape.len() {
                1 => vec! [1, 1, 1, shape[0]],
                2 => vec! [shape[0], 1, 1, shape[1]],
                4 => shape,
                _ => { unreachable!() }
            };

            if !tensor_by_shape.contains_key(&shape) {
                tensor_by_shape.insert(shape.clone(), tensor_factory::get_or_create(
                    cudnn::cudnnDataType_t::Half,
                    cudnn::cudnnTensorFormat_t::NHWC,
                    &shape
                )?);
            }

            prepared_session.tensors.insert(variable_def.id, tensor_by_shape[&shape].clone());
        }

        for (layer, layer_def) in graph.layers() {
            let prepared_layer = layer.prepare(
                &handle,
                &prepared_session.tensors(&layer_def.input),
                &prepared_session.tensors(&layer_def.output)
            )?;

            prepared_session.layers.push((prepared_layer, layer_def.clone()));
        }

        Ok(prepared_session)
    }

    fn compile(
        &mut self,
        handle: &mut cudnn::Handle,
        variables: &FnvHashMap<usize, Arc<cuda::Ptr>>,
        workspace: &cuda::Ptr
    ) -> Result<(), cuda::Error>
    {
        let stream = cuda::Stream::new()?;
        let stream_capture = stream.begin_capture()?;

        handle.set_stream(&stream)?;
        for (layer, layer_def) in &self.layers {
            let inputs: Vec<(&cudnn::Tensor, *const c_void)> = layer_def.input.iter()
                .map(|variable_def| {
                    let id = variable_def.id;

                    (self.tensors[&id].as_ref(),variables[&id].as_ptr() as *const c_void)
                })
                .collect();
            let outputs: Vec<(&cudnn::Tensor, *mut c_void)> = layer_def.output.iter()
                .map(|variable_def| {
                    let id = variable_def.id;

                    (self.tensors[&id].as_ref(), variables[&id].as_ptr())
                })
                .collect();

            layer.forward(
                handle,
                &inputs,
                &outputs,
                workspace.as_ptr()
            )?;
        }

        let graph = stream_capture.end_capture()?;
        let graph_exec = graph.as_graph_exec()?;

        self.graph = Some(graph);
        self.graph_exec = Some(graph_exec);

        Ok(())
    }

    fn forward(
        &mut self,
        handle: &cudnn::Handle
    ) -> Result<(), cuda::Error>
    {
        let graph_exec = self.graph_exec.as_ref().unwrap();
        let stream = handle.get_stream()?;

        graph_exec.launch(&stream)
    }

    fn size_in_bytes(&self) -> usize {
        self.layers.iter()
            .map(|(layer, _layer_def)| layer.size_in_bytes())
            .max()
            .unwrap_or(0)
    }

    fn tensors(&self, variable_defs: &[VariableDef]) -> Vec<&cudnn::Tensor> {
        variable_defs.iter()
            .filter_map(|variable_def| self.tensors.get(&variable_def.id))
            .map(|tensor| tensor.as_ref())
            .collect()
    }
}

impl Session {
    pub fn new(graph: &Arc<Graph>, max_batch_size: usize) -> Result<Session, cuda::Error> {
        debug_assert!(graph.device.is_current()?);

        let mut out = Session {
            graph: graph.clone(),
            variables: FnvHashMap::default(),
            workspace: cuda::Ptr::null(),
            handle: cudnn::Handle::new()?,
            stream: cuda::Stream::new()?,
            by_batch_size: Vec::with_capacity(max_batch_size)
        };

        // prepare one set of tensors and prepared layers for each possible
        // batch size, this is not as expensive as it sounds because the batch
        // sizes share weights, workspaces, and input and output pointers. So we
        // only really duplicate some `cudnnTensorDescriptor_t`, which is not that
        // expensive.
        let mut max_size_in_bytes = 0;

        for batch_size in 1..=max_batch_size {
            let prepared_session = PreparedSession::new(&out.handle, graph, batch_size)?;

            max_size_in_bytes = ::std::cmp::max(
                prepared_session.size_in_bytes(),
                max_size_in_bytes
            );
            out.by_batch_size.push(prepared_session);
        }

        // allocate the maximum workspace needed by any batch size, and the
        // registers.
        out.workspace = cuda::Ptr::new(max_size_in_bytes)?;
        out.allocate_variables(&graph, max_batch_size)?;

        // for each batch size, record a graph stream of the inference
        for prepared_session in out.by_batch_size.iter_mut() {
            prepared_session.compile(
                &mut out.handle,
                &out.variables,
                &out.workspace
            )?;
        }

        out.handle.set_stream(&out.stream)?;

        Ok(out)
    }

    /// Greedy register allocation of all variables based on a _random_
    /// ordering, which seems to converge to an optimal solution most of the
    /// time anyway.
    ///
    /// # Arguments
    ///
    /// * `graph` -
    /// * `max_batch_size` -
    ///
    fn allocate_variables(
        &mut self,
        graph: &Graph,
        max_batch_size: usize
    ) -> Result<(), cuda::Error>
    {
        let active_variables = alive_variables(graph);
        let num_variables = graph.variables().len();
        let mut colored = FnvHashMap::default();
        let mut by_color = FnvHashMap::default();

        for variable_def in graph.variables().values() {
            let mut neighbours = FnvHashSet::default();

            for actives in &active_variables {
                if actives.contains(&variable_def.id) {
                    for neighbour in actives {
                        if colored.contains_key(neighbour) {
                            neighbours.insert(colored[neighbour]);
                        }
                    }
                }
            }

            let coloring = (0..num_variables)
                .filter(|c| !neighbours.contains(c))
                .next()
                .unwrap();

            colored.insert(variable_def.id, coloring);
            by_color.entry(coloring)
                .or_insert_with(|| FnvHashSet::default())
                .insert(variable_def.id);
        }

        for (_color, variable_ids) in by_color.into_iter() {
            let max_size = variable_ids.iter()
                .map(|id| graph.variables()[id].size())
                .max()
                .unwrap_or(0);
            let size_in_bytes = max_batch_size * max_size * cudnn::cudnnDataType_t::Half.size_in_bytes();
            let ptr = Arc::new(cuda::Ptr::new(size_in_bytes)?);

            for variable_id in variable_ids {
                self.variables.insert(variable_id, ptr.clone());
            }
        }

        // check that all
        for variable_def in graph.variables().values() {
            assert!(self.variables.contains_key(&variable_def.id));
        }

        Ok(())
    }

    pub fn forward(
        &mut self,
        inputs: &HashMap<String, &[f16]>,
        batch_size: usize
    ) -> Result<HashMap<String, Vec<f16>>, cuda::Error>
    {
        debug_assert!(batch_size > 0);

        // copy the provided inputs to the tensors
        for (name, input_def) in self.graph.inputs() {
            assert!(inputs.contains_key(name));

            let ptr = &self.variables[&input_def.id];
            let src = inputs[name];

            cuda::copy_nonoverlapping(
                src.as_ptr(),
                ptr.as_ptr() as *mut f16,
                src.len(),
                cuda::cudaMemcpyKind_t::HostToDevice,
                &self.stream
            )?;
        }

        // execute all of the stored procedures
        self.by_batch_size[batch_size - 1].forward(&self.handle)?;

        // copy all results back to the host, and return them
        let mut outputs = HashMap::default();

        for (name, output_def) in self.graph.outputs() {
            let mut output_data = vec! [f16::from(0.0); batch_size * output_def.size()];
            let ptr = &self.variables[&output_def.id];

            cuda::copy_nonoverlapping(
                ptr.as_ptr() as *const f16,
                output_data.as_mut_ptr(),
                output_data.len(),
                cuda::cudaMemcpyKind_t::DeviceToHost,
                &self.stream
            )?;
            outputs.insert(name.clone(), output_data);
        }
        self.stream.synchronize()?;

        Ok(outputs)
    }
}

fn alive_variables(graph: &Graph) -> Vec<FnvHashSet<usize>> {
    // Do a liveness analysis to determine which variables are alive at each
    // point using a two phase algorithm:
    //
    //   (i) collect all variables that has been observed so far using
    //       a forward pass.
    //
    let layers = graph.layers();
    let mut live_variables: Vec<FnvHashSet<usize>> = vec! [];

    for (_layer, layer_def) in layers {
        let mut actives = match live_variables.last() {
            None => FnvHashSet::default(),
            Some(x) => x.clone()
        };

        for variable_def in layer_def.variables() {
            actives.insert(variable_def.id);
        }

        live_variables.push(actives);
    }

    //
    //   (ii) in a reverse pass, collect all variables that that has been
    //        observed and only retain those in the sets build from the
    //        first phase.
    //
    let mut retained_variables = FnvHashSet::default();

    for (_name, output_def) in graph.outputs() {
        retained_variables.insert(output_def.id);
    }

    for (i, (_layer, layer_def)) in layers.iter().enumerate().rev() {
        for variable_def in layer_def.variables() {
            retained_variables.insert(variable_def.id);
        }

        live_variables[i].retain(|val| retained_variables.contains(val));
    }

    live_variables
}

#[cfg(test)]
mod tests {
    // pass
}