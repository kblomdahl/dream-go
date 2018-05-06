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

use std::collections::HashMap;
use std::env;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::{Arc, Mutex};

use nn::devices::get_current_device;
use nn::graph;
use nn::loader;

type WorkspaceQueue = Mutex<Vec<graph::Workspace>>;

/// Wrapper around a `Workspace` that when dropped returns it to the
/// pool it was acquired from.
pub struct WorkspaceGuard<'a> {
    workspace: Option<graph::Workspace>,
    pool: *mut WorkspaceQueue,

    lifetime: ::std::marker::PhantomData<&'a ()>
}

impl<'a> Deref for WorkspaceGuard<'a> {
    type Target = graph::Workspace;

    fn deref(&self) -> &Self::Target { self.workspace.as_ref().unwrap() }
}

impl<'a> DerefMut for WorkspaceGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target { self.workspace.as_mut().unwrap() }
}

impl<'a> Drop for WorkspaceGuard<'a> {
    fn drop(&mut self) {
        unsafe {
            let workspace = self.workspace.take().unwrap();
            let mut pool = (*self.pool).lock().unwrap();

            pool.push(workspace);
        }
    }
}

/// Pool of workspaces that can be used for network evaluations.
#[derive(Clone)]
pub struct Network {
    builder: Arc<graph::Builder>,
    workspaces: Arc<Mutex<HashMap<(usize, i32), Box<WorkspaceQueue>>>>
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
                }).unwrap_or("dream_go.json".to_string()),

                // hard-coded paths to make development and deployment easier
                "dream_go.json".to_string(),
                "models/dream_go.json".to_string(),
                "/usr/share/dreamgo/dream_go.json".to_string(),
                "/usr/share/dream_go/dream_go.json".to_string()
            ];
        }

        PATHS.iter()
            .filter_map(|path| loader::load(Path::new(path)))
            .next()
            .map(|weights| Network {
                builder: Arc::new(graph::Builder::new(weights)),
                workspaces: Arc::new(Mutex::new(HashMap::new()))
            })
    }

    /// Returns a `Workspace` with the given batch size.
    /// 
    /// # Arguments
    /// 
    /// * `batch_size` -
    /// 
    pub fn get_workspace<'a>(&'a self, batch_size: usize) -> WorkspaceGuard<'a> {
        let device_id = get_current_device();
        let key = (batch_size, device_id);
        let mut workspaces = self.workspaces.lock().unwrap();
        let candidates = workspaces.entry(key).or_insert_with(|| Box::new(Mutex::new(vec! [])));
        let candidates_ptr = &mut **candidates as *mut WorkspaceQueue;
        let guard = match candidates.lock().unwrap().pop() {
            Some(workspace) => WorkspaceGuard {
                workspace: Some(workspace),
                pool: candidates_ptr,
                lifetime: ::std::marker::PhantomData::default()
            },
            None => WorkspaceGuard {
                workspace: Some(self.builder.get_workspace(batch_size)),
                pool: candidates_ptr,
                lifetime: ::std::marker::PhantomData::default()
            }
        };

        guard
    }
}
