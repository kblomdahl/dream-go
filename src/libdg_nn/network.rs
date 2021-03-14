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

use crossbeam_channel::{self, Sender, Receiver};
use dashmap::DashMap;
use std::env;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::Arc;

use dg_cuda::{Device, PerDevice};

use super::{Error, graph, loader};

#[derive(Clone)]
struct WorkspaceQueue {
    tx: Sender<graph::Workspace>,
    rx: Receiver<graph::Workspace>
}

impl Default for WorkspaceQueue {
    fn default() -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();

        Self {
            tx, rx
        }
    }
}

impl WorkspaceQueue {
    pub fn push(&self, workspace: graph::Workspace) {
        self.tx.send(workspace).expect("could not push `Workspace` to queue");
    }

    pub fn try_pop(&self) -> Option<graph::Workspace> {
        self.rx.try_recv().ok()
    }
}

/// Wrapper around a `Workspace` that when dropped returns it to the
/// pool it was acquired from.
pub struct WorkspaceGuard {
    workspace: Option<graph::Workspace>,
    pool: WorkspaceQueue
}

impl Deref for WorkspaceGuard {
    type Target = graph::Workspace;

    fn deref(&self) -> &Self::Target {
        self.workspace.as_ref().unwrap()
    }
}

impl DerefMut for WorkspaceGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.workspace.as_mut().unwrap()
    }
}

impl Drop for WorkspaceGuard {
    fn drop(&mut self) {
        if let Some(workspace) = self.workspace.take() {
            self.pool.push(workspace);
        }
    }
}

/// Pool of workspaces that can be used for network evaluations.
#[derive(Clone)]
pub struct Network {
    builder: Arc<graph::Builder>,
    workspaces: Arc<PerDevice<DashMap<usize, WorkspaceQueue>>>
}

unsafe impl Send for Network { }  // this is safe because the Rc<...> is guarded by a Mutex and/or Arc
unsafe impl Sync for Network { }  // this is safe because the Rc<...> is guarded by a Mutex and/or Arc

impl Network {
    pub fn new() -> Option<Network> {
        lazy_static! {
            static ref PATHS: Vec<String> = vec! [
                // check for a file named the same as the current executable, but
                // with a `.json` extension.
                env::current_exe().ok().and_then(|file| {
                    let mut json = file.clone();
                    json.set_extension("json");
                    json.as_path().to_str().map(|s| s.to_string())
                }).unwrap_or_else(|| "dream_go.json".to_string()),

                // hard-coded paths to make development and deployment easier
                "dream_go.json".to_string(),
                "models/dream_go.json".to_string(),
                "/usr/share/dreamgo/dream_go.json".to_string(),
                "/usr/share/dream_go/dream_go.json".to_string()
            ];
        }

        PATHS.iter()
            .filter_map(|path| {
                match loader::load(Path::new(path)) {
                    Ok(weights) => Some(weights),
                    Err(Error::MissingWeights) => None,
                    Err(reason) => {
                        panic!("Failed to load network weights -- {:?}", reason)
                    }
                }
            })
            .next()
            .map(|weights| Network {
                builder: Arc::new(graph::Builder::new(weights)),
                workspaces: Arc::new(PerDevice::new().expect("could not create PerDevice<T>"))
            })
    }

    /// Returns a `Workspace` with the given batch size.
    /// 
    /// # Arguments
    /// 
    /// * `batch_size` -
    /// 
    pub fn get_workspace(&self, batch_size: usize) -> Result<WorkspaceGuard, Error> {
        let candidates = self.workspaces.entry(batch_size).or_default();

        Ok(WorkspaceGuard {
            pool: candidates.clone(),
            workspace: match candidates.try_pop() {
                None => Some(self.builder.get_workspace(batch_size)?),
                x => x
            }
        })
    }

    /// Wait for all jobs on the current device to finish, and then drain all of the workspaces.
    pub fn synchronize(&self) {
        let original_device = Device::default();

        for device in Device::all().unwrap() {
            device.set_current().expect("Failed to set the device for the current thread");
            let status = Device::synchronize();  // this should be allowed to fail
            if let Err(err) = status {
                eprintln!("Error: {:?}", err);
            }

            self.workspaces.clear();
        }

        original_device.set_current().expect("Failed to set the device for the current thread");
    }
}
