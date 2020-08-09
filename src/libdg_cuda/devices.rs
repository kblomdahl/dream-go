// Copyright 2020 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use libc::{c_int};

use crate::{Error, cudaError_t};

#[repr(i32)]
#[derive(Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum cudaDeviceAttr {
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76
}

#[link(name = "cuda")]
#[link(name = "cudart")]
extern {
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut c_int) -> cudaError_t;
    pub fn cudaSetDevice(device: c_int) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaRuntimeGetVersion(version: *mut c_int) -> cudaError_t;
    pub fn cudaDeviceGetAttribute(value: *mut c_int, attr: cudaDeviceAttr, device: c_int) -> cudaError_t;
}

pub struct Device {
    device_id: i32
}

impl Default for Device {
    fn default() -> Self {
        let mut device_id: i32 = 0;
        unsafe { cudaGetDevice(&mut device_id); }
        Self { device_id }
    }
}

impl Device {
    pub fn new(device_id: i32) -> Self {
        Self { device_id }
    }

    pub fn len() -> Result<i32, Error> {
        let mut device_count = 0;
        let status = unsafe { cudaGetDeviceCount(&mut device_count) };
        status.into_result(device_count)
    }

    pub fn all() -> Result<Vec<Device>, Error> {
        let device_count = Self::len()?;
        Ok((0..device_count).map(|i| Device::new(i)).collect())
    }

    pub fn id(&self) -> i32 {
        self.device_id
    }

    pub fn set_current(&self) -> Result<(), Error> {
        let status = unsafe { cudaSetDevice(self.device_id) };
        status.into_result(())
    }

    pub fn synchronize() -> Result<(), Error> {
        let status = unsafe { cudaDeviceSynchronize() };
        status.into_result(())
    }

    pub fn compute_capability(&self) -> Result<(i32, i32), Error> {
        let mut version_major: i32 = 0;
        let mut version_minor: i32 = 0;

        unsafe {
            cudaDeviceGetAttribute(&mut version_major, cudaDeviceAttr::ComputeCapabilityMajor, self.device_id).into_result(())?;
            cudaDeviceGetAttribute(&mut version_minor, cudaDeviceAttr::ComputeCapabilityMinor, self.device_id).into_result(())?;
        }

        Ok((version_major, version_minor))
    }

    pub fn is_supported(&self) -> Result<bool, Error> {
        let (major_version, _minor_version) = self.compute_capability()?;

        Ok(major_version >= 7)
    }
}

#[cfg(test)]
mod tests {
    // pass
}
