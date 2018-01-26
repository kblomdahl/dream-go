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

use libc::{c_void};

#[macro_use] pub mod ffi;
mod graph;
mod loader;
mod network;
mod ops;

use self::ffi::cublas::*;
use self::ffi::cuda::*;
use self::ffi::cudnn::*;
use self::graph::Graph;
pub use self::graph::Workspace;
pub use self::network::{Network, WorkspaceGuard};

/// Returns the value and policy tensors obtained from a forward pass
/// through the neural network.
///
/// # Arguments
///
/// * `ws` - the workspace for the current thread
/// * `features` - the input features
///
pub fn forward<T: From<f32> + Clone, R: From<f32> + Clone>(
    workspace: &mut Workspace,
    features: &Vec<Box<[T]>>
) -> (Vec<R>, Vec<Box<[R]>>)
{
    let batch_size = workspace.batch_size;

    debug_assert!(batch_size == features.len());

    let mut softmax = vec! [vec! [R::from(0.0f32); 362]; batch_size];
    let mut value = vec! [R::from(0.0f32); batch_size];

    unsafe {
        check!(cudnnSetStream(workspace.handle_dnn, workspace.tower_stream));
        check!(cublasSetStream_v2(workspace.handle_blas, workspace.tower_stream));

        for (i, ref feature) in features.iter().enumerate() {
            assert_eq!(feature.len(), 11552);
            assert_eq!(1, ::std::mem::size_of::<c_void>());

            let element_size = ::std::mem::size_of::<T>() * 11552;
            let input = workspace.get_input(None)
                .offset((i * element_size) as isize);

            check!(cudaMemcpyAsync(
                input,
                feature.as_ptr() as *const c_void,
                element_size,
                MemcpyKind::HostToDevice,
                workspace.tower_stream
            ));
        }

        graph::tower::<graph::Runtime, _>(workspace);

        // fix the synchronization, so that the policy and value head needs to
        // wait for the tower to complete before they can start.
        check!(cudaEventRecord(workspace.tower_event, workspace.tower_stream));

        check!(cudaStreamWaitEvent(workspace.policy_stream, workspace.tower_event, 0));
        check!(cudaStreamWaitEvent(workspace.value_stream, workspace.tower_event, 0));

        // policy head (21p_policy)
        check!(cudnnSetStream(workspace.handle_dnn, workspace.policy_stream));
        check!(cublasSetStream_v2(workspace.handle_blas, workspace.policy_stream));

        graph::policy::<graph::Runtime, _>(workspace);

        for i in 0..batch_size {
            let element_size = ::std::mem::size_of::<R>() * 362;
            let output = workspace.get_policy_output(None)
                .offset((i * element_size) as isize);

            check!(cudaMemcpyAsync(
                softmax[i].as_mut_ptr() as *mut c_void,
                output,
                element_size,
                MemcpyKind::DeviceToHost,
                workspace.policy_stream
            ));
        }

        // value head (21v_value)
        check!(cudnnSetStream(workspace.handle_dnn, workspace.value_stream));
        check!(cublasSetStream_v2(workspace.handle_blas, workspace.value_stream));

        graph::value::<graph::Runtime, _>(workspace);

        check!(cudaMemcpyAsync(
            value.as_mut_ptr() as *mut c_void,
            workspace.get_value_output(None),
            batch_size * ::std::mem::size_of::<R>(),
            MemcpyKind::DeviceToHost,
            workspace.value_stream
        ));

        // wait for both the value and policy head to finish
        check!(cudaStreamSynchronize(workspace.policy_stream));
        check!(cudaStreamSynchronize(workspace.value_stream));
    }

    (value, softmax.into_iter().map(|s| s.into_boxed_slice()).collect())
}
