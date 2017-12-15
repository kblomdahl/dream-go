// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use nn::ffi::cuda::*;
use nn::ffi::cudnn::DataType;
use nn::workspace::{Shared, Workspace};

type WorkspaceQueue = Rc<RefCell<Vec<Rc<Workspace>>>>;

/// Wrapper around a `Workspace` that when dropped returns it to the
/// pool it was acquired from.
pub struct WorkspaceGuard<'a> {
    workspace: Rc<Workspace>,
    pool: &'a Mutex<HashMap<usize, WorkspaceQueue>>
}

impl<'a> Deref for WorkspaceGuard<'a> {
    type Target = Workspace;

    fn deref(&self) -> &Self::Target { &self.workspace }
}

impl<'a> DerefMut for WorkspaceGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target { Rc::get_mut(&mut self.workspace).unwrap() }
}

impl<'a> Drop for WorkspaceGuard<'a> {
    fn drop(&mut self) {
        let workspace = self.workspace.clone();
        let batch_size = workspace.batch_size;
        let workspaces = self.pool.lock().unwrap();
        let mut candidates = workspaces[&batch_size].borrow_mut();

        candidates.push(workspace);
    }
}

/// Returns the version of the CUDA Runtime library.
fn runtime_version() -> i32 {
    let mut runtime_version: i32 = 0;

    unsafe {
        check!(cudaRuntimeGetVersion(&mut runtime_version));
    }

    runtime_version
}

/// Returns the major and minor version (in that order) of the CUDA
/// Compute Capability for the currently selected device.
fn compute_capability() -> (i32, i32) {
    let mut version_major: i32 = 0;
    let mut version_minor: i32 = 0;

    unsafe {
        check!(cudaDeviceGetAttribute(&mut version_major, DeviceAttr::ComputeCapabilityMajor, 0));
        check!(cudaDeviceGetAttribute(&mut version_minor, DeviceAttr::ComputeCapabilityMinor, 0));
    }

    (version_major, version_minor)
}

/// Pool of workspaces that can be used for network evaluations.
pub struct Network {
    shared: Arc<Shared>,
    workspaces: Mutex<HashMap<usize, WorkspaceQueue>>
}

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
                }).unwrap_or("dream_go.json".to_string()),

                // hard-coded paths to make development and deployment easier
                "dream_go.json".to_string(),
                "models/dream_go.json".to_string(),
                "/usr/share/dream_go/dream_go.json".to_string()
            ];
        }

        // figure out if it is better to use half precision or single precision
        let data_type = if runtime_version() >= 9000 && compute_capability().0 >= 7 {
            DataType::Half
        } else {
            DataType::Float
        };

        PATHS.iter()
            .filter_map(|path| Shared::new(Path::new(path), data_type))
            .next()
            .map(|shared| Network { shared: Arc::new(shared), workspaces: Mutex::new(HashMap::new()) })
    }

    /// Returns a `Workspace` with the given batch size.
    /// 
    /// # Arguments
    /// 
    /// * `batch_size` -
    /// 
    pub fn get_workspace<'a>(&'a self, batch_size: usize) -> WorkspaceGuard<'a> {
        let mut workspaces = self.workspaces.lock().unwrap();
        let candidates = workspaces.entry(batch_size).or_insert_with(|| Rc::new(RefCell::new(vec! [])));
        let guard = match candidates.borrow_mut().pop() {
            Some(workspace) => WorkspaceGuard {
                workspace: workspace.clone(),
                pool: &self.workspaces
            },
            None => WorkspaceGuard {
                workspace: Rc::new(Workspace::new(&self.shared, batch_size)),
                pool: &self.workspaces
            }
        };

        guard
    }
}

impl Deref for Network {
    type Target = Shared;

    fn deref(&self) -> &Self::Target { &self.shared }
}
