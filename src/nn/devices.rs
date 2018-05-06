// Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use nn::ffi::cuda;

/// The maximum number of devices that we support. 
pub const MAX_DEVICES: usize = 8;

lazy_static! {
    pub static ref DEVICES: Vec<i32> = {
        let devices: Vec<i32> = unsafe {
            let mut count: i32 = 0;

            check!(cuda::cudaGetDeviceCount(&mut count));

            (0..count).filter(|&device_id| is_supported(device_id)).collect()
        };

        if devices.is_empty() {
            panic!("No device available with the required compute capacity (6.1)");
        }

        devices
    };
}

pub fn get_current_device() -> i32 {
    unsafe {
        let mut device_id: i32 = 0;

        check!(cuda::cudaGetDevice(&mut device_id));
        device_id
    }
}

pub fn set_current_device(device_id: i32) {
    unsafe {
        check!(cuda::cudaSetDevice(device_id));
    }
}

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
fn compute_capability(device_id: i32) -> (i32, i32) {
    let mut version_major: i32 = 0;
    let mut version_minor: i32 = 0;

    unsafe {
        check!(cuda::cudaDeviceGetAttribute(&mut version_major, cuda::DeviceAttr::ComputeCapabilityMajor, device_id));
        check!(cuda::cudaDeviceGetAttribute(&mut version_minor, cuda::DeviceAttr::ComputeCapabilityMinor, device_id));
    }

    (version_major, version_minor)
}

/// Returns whether we should use DP4A on the current device.
/// 
/// There is no flag that NVIDIA expose to determine this, so we
/// determine this by the CUDA version (>= 8) and the compute
/// capabilities (6.1+).
pub fn is_supported(device_id: i32) -> bool {
    let (major, minor) = compute_capability(device_id);
    let version = runtime_version();

    version >= 8000 && (major == 6 && minor >= 1 || major >= 7)
}
