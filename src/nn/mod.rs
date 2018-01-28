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

use self::ffi::cublas;
use self::ffi::cuda;
use self::ffi::cudnn;
use self::graph::Graph;
pub use self::graph::Workspace;
pub use self::network::{Network, WorkspaceGuard};

/// Returns the version of the CUDA Runtime library.
fn runtime_version() -> i32 {
    let mut runtime_version: i32 = 0;

    unsafe {
        check!(cuda::cudaRuntimeGetVersion(&mut runtime_version));
    }

    runtime_version
}

/// Returns the major and minor version (in that order) of the CUDA
/// Compute Capability for the currently selected device.
fn compute_capability() -> (i32, i32) {
    let mut version_major: i32 = 0;
    let mut version_minor: i32 = 0;

    unsafe {
        check!(cuda::cudaDeviceGetAttribute(&mut version_major, cuda::DeviceAttr::ComputeCapabilityMajor, 0));
        check!(cuda::cudaDeviceGetAttribute(&mut version_minor, cuda::DeviceAttr::ComputeCapabilityMinor, 0));
    }

    (version_major, version_minor)
}

/// Returns whether we should use half precision on the current device.
/// 
/// See the (CUDA Programmers Guide)[http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions]
/// for an exhaustive list of what each compute capability means.
fn should_use_half() -> bool {
    let (major, minor) = compute_capability();

    major == 6 && (minor == 0 || minor == 2) ||
    major == 7 && minor == 0
}

/// Returns whether we should use tensor cores on the current device.
/// 
/// There is no flag that NVIDIA expose to determine this, so we
/// determine this by the CUDA version (>= 9) and the compute
/// capabilities (7.0).
fn should_use_tensor_core() -> bool {
    if cfg!(feature = "tensor-core") {
        let (major, minor) = compute_capability();
        let version = runtime_version();

        version >= 9000 && major == 7 && minor == 0
    } else {
        false
    }
}

/// Returns whether we should use DP4A on the current device.
/// 
/// There is no flag that NVIDIA expose to determine this, so we
/// determine this by the CUDA version (>= 8) and the compute
/// capabilities (6.1+).
fn should_use_dp4a() -> bool {
    if cfg!(feature = "dp4a") {
        let (major, minor) = compute_capability();
        let version = runtime_version();

        version >= 8000 && (major == 6 && minor >= 1 || major >= 7)
    } else {
        false
    }
}

/// The supported types for inference.
pub enum Type {
    Int8,
    Half,
    Single
}

lazy_static! {
    /// The type that is expected as input to `forward`. This is determined
    /// based on the _Compute Capability_ of the users GPU.
    pub static ref TYPE: Type = if should_use_half() || should_use_tensor_core() {
        Type::Half
    } else if should_use_dp4a() {
        Type::Int8
    } else {
        Type::Single
    };
}

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
        check!(cudnn::cudnnSetStream(workspace.handle_dnn, workspace.tower_stream));
        check!(cublas::cublasSetStream_v2(workspace.handle_blas, workspace.tower_stream));

        for (i, ref feature) in features.iter().enumerate() {
            assert_eq!(feature.len(), 11552);
            assert_eq!(1, ::std::mem::size_of::<c_void>());

            let element_size = ::std::mem::size_of::<T>() * 11552;
            let input = workspace.get_input(None)
                .offset((i * element_size) as isize);

            check!(cuda::cudaMemcpyAsync(
                input,
                feature.as_ptr() as *const c_void,
                element_size,
                cuda::MemcpyKind::HostToDevice,
                workspace.tower_stream
            ));
        }

        graph::tower::<graph::Runtime, _>(workspace);

        // fix the synchronization, so that the policy and value head needs to
        // wait for the tower to complete before they can start.
        check!(cuda::cudaEventRecord(workspace.tower_event, workspace.tower_stream));

        check!(cuda::cudaStreamWaitEvent(workspace.policy_stream, workspace.tower_event, 0));
        check!(cuda::cudaStreamWaitEvent(workspace.value_stream, workspace.tower_event, 0));

        // policy head (21p_policy)
        check!(cudnn::cudnnSetStream(workspace.handle_dnn, workspace.policy_stream));
        check!(cublas::cublasSetStream_v2(workspace.handle_blas, workspace.policy_stream));

        graph::policy::<graph::Runtime, _>(workspace);

        for i in 0..batch_size {
            let element_size = ::std::mem::size_of::<R>() * 362;
            let output = workspace.get_policy_output(None)
                .offset((i * element_size) as isize);

            check!(cuda::cudaMemcpyAsync(
                softmax[i].as_mut_ptr() as *mut c_void,
                output,
                element_size,
                cuda::MemcpyKind::DeviceToHost,
                workspace.policy_stream
            ));
        }

        // value head (21v_value)
        check!(cudnn::cudnnSetStream(workspace.handle_dnn, workspace.value_stream));
        check!(cublas::cublasSetStream_v2(workspace.handle_blas, workspace.value_stream));

        graph::value::<graph::Runtime, _>(workspace);

        check!(cuda::cudaMemcpyAsync(
            value.as_mut_ptr() as *mut c_void,
            workspace.get_value_output(None),
            batch_size * ::std::mem::size_of::<R>(),
            cuda::MemcpyKind::DeviceToHost,
            workspace.value_stream
        ));

        // wait for both the value and policy head to finish
        check!(cuda::cudaStreamSynchronize(workspace.policy_stream));
        check!(cuda::cudaStreamSynchronize(workspace.value_stream));
    }

    (value, softmax.into_iter().map(|s| s.into_boxed_slice()).collect())
}
