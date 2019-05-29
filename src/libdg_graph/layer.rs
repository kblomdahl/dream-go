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

use std::hash::{Hash, Hasher};
use std::ops::Deref;

use libc::c_void;

use dg_cuda as cuda;
use dg_cuda::cudnn;
use graph_def::LayerDef;

pub trait Layer : Sync + Send {
    /// Prepare any `inputs` or `outputs` specific parts of the layer, and return a
    /// structure containing them.
    ///
    /// # Arguments
    ///
    /// * `handle` -
    /// * `inputs` -
    /// * `outputs` -
    ///
    fn prepare(
        &self,
        handle: &cudnn::Handle,
        inputs: &[&cudnn::Tensor],
        outputs: &[&cudnn::Tensor]
    ) -> Result<Box<PreparedLayer>, cuda::Error>;
}

pub trait PreparedLayer : Sync + Send {
    /// Returns the size of the necessary workspace in bytes.
    fn size_in_bytes(&self) -> usize;

    /// Perform the forward pass of this layer, for the given `inputs` and `outputs`. If
    /// a workspace is necessary then a `workspace_ptr` of at least the required size is
    /// provided.
    ///
    /// # Arguments
    ///
    /// * `handle` -
    /// * `inputs` -
    /// * `outputs` -
    /// * `workspace_ptr` -
    ///
    fn forward(
        &self,
        handle: &cudnn::Handle,
        inputs: &[(&cudnn::Tensor, *const c_void)],
        outputs: &[(&cudnn::Tensor, *mut c_void)],
        workspace_ptr: *mut c_void
    ) -> Result<(), cuda::Error>;
}

pub struct LayerId<'a> {
    pub id: usize,
    pub layer_def: &'a LayerDef
}

impl<'a> Deref for LayerId<'a> {
    type Target = LayerDef;

    fn deref(&self) -> &Self::Target {
        self.layer_def
    }
}

impl<'a> Hash for LayerId<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.id)
    }
}

impl<'a> PartialEq for LayerId<'a> {
    fn eq(&self, other: &LayerId) -> bool {
        self.id == other.id
    }
}

impl<'a> Eq for LayerId<'a> {
    // pass
}