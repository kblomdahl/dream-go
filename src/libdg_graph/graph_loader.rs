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

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use fnv::FnvHashMap;

use dg_cuda as cuda;
use dg_cuda::Device;
use dg_utils::config;
use graph::Graph;
use graph_def::GraphDef;
use optimizers::{RemoveRedundantIdentityLayers, ScaleBeforeSoftmax, TransformInputOutput};
use Session;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub enum GraphLoaderError {
    NotFound,
    InvalidGraph(String, serde_json::Error)
}

pub struct GraphLoader {
    graph_def: Arc<GraphDef>,
    by_device: Mutex<FnvHashMap<Device, Arc<Graph>>>
}

impl GraphLoader {
    pub fn new() -> Result<GraphLoader, GraphLoaderError> {
        let buf_rd = open_graph_file();

        if let Some((buf_rd, path)) = buf_rd {
            let graph_def = match GraphDef::from_reader(buf_rd) {
                Err(err) => { return Err(GraphLoaderError::InvalidGraph(path, err)) },
                Ok(graph_def) => graph_def,
            };

            Ok(GraphLoader {
                graph_def: Arc::new(optimize(graph_def)),
                by_device: Mutex::new(FnvHashMap::default())
            })
        } else {
            Err(GraphLoaderError::NotFound)
        }
    }

    /// Create a new `Session` for the current thread and the provided arguments.
    ///
    /// # Arguments
    ///
    /// * `outputs` -
    /// * `max_batch_size` -
    ///
    pub fn create_session(
        &self,
        outputs: &[String],
        max_batch_size: usize
    ) -> Result<Session, cuda::Error>
    {
        let current_device = cuda::Device::current()?;
        let mut by_device = self.by_device.lock().unwrap();

        if !by_device.contains_key(&current_device) {
            by_device.insert(current_device, Arc::new(Graph::new(
                &self.graph_def,
                current_device,
                outputs
            )?));
        }

        Session::new( &by_device[&current_device], max_batch_size)
    }

    /// Try to recover from any error by resetting the internal CUDA state.
    pub fn synchronize(&mut self) -> Result<(), cuda::Error> {
        let mut by_device = self.by_device.lock().unwrap();
        by_device.clear();

        Ok(())
    }
}

fn open_graph_file() -> Option<(impl BufRead, String)> {
    let paths: Vec<String> = vec! [
        // check for a file named the same as the current executable, but
        // with a `.json` extension.
        env::current_exe().ok().and_then(|file| {
            let mut json = file.clone();
            json.set_extension("json");
            json.as_path().to_str().map(|s| s.to_string())
        }).unwrap_or_else(|| "dream_go.json".to_string()),

        // hard-coded paths to make development and deployment easier
        "dream_go.json".to_string(),
        "/usr/share/dreamgo/dream_go.json".to_string(),
        "/usr/share/dream_go/dream_go.json".to_string()
    ];

    paths.iter()
        .filter_map(|path| {
            let os_path = Path::new(path);

            match File::open(os_path) {
                Ok(f) => Some((BufReader::new(f), path.clone())),
                Err(_) => None
            }
        })
        .next()
}

fn optimize(g: GraphDef) -> GraphDef {
    let g = if *config::SOFTMAX_TEMPERATURE != 1.0 {
        ScaleBeforeSoftmax::apply(g, *config::SOFTMAX_TEMPERATURE)
    } else {
        g
    };

    let g = TransformInputOutput::apply(g);
    let g = RemoveRedundantIdentityLayers::apply(g);

    g
}