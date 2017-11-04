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

//mod cudnn;  // ffi
//mod cuda;  // ffi

/// A per-thread workspace containing tensor descriptors, operations, and
/// data arrays necessary to perform a forward pass through the neural
/// network.
pub struct Workspace {
    // pass
}

impl Workspace {
    fn new() -> Workspace {
        Workspace {
            // pass
        }
    }
}

/// Returns the value and policy tensors obtained from a forward pass
/// through the neural network.
///
/// # Arguments
///
/// * `ws` - the workspace for the current thread
/// * `features` - the input features
///
fn forward(ws: &mut Workspace, features: &[f32]) -> (f32, Box<[f32]>) {
    (0.0, vec! [].into_boxed_slice())
}
