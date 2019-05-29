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

use super::error::Error;

use libc::c_int;


#[repr(i32)]
pub enum CudaDeviceAttr {
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76
}

#[link(name = "cuda")]
#[link(name = "cudart")]
extern {
    fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
    fn cudaGetDevice(device: *mut c_int) -> c_int;
    fn cudaSetDevice(device: c_int) -> c_int;

    fn cudaDeviceSynchronize() -> c_int;
    fn cudaDeviceGetAttribute(value: *mut c_int, attr: CudaDeviceAttr, device: c_int) -> c_int;
}

lazy_static! {
    pub static ref NUM_DEVICES: Result<c_int, Error> = {
        let mut num_devices = 0;
        let success = unsafe { cudaGetDeviceCount(&mut num_devices) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(num_devices)
        }
    };

    pub static ref DEVICES: Result<Vec<Device>, Error> = {
        NUM_DEVICES.clone().map(|num_devices| {
            (0..num_devices)
                .map(|i| Device(i))
                .filter(|device| {
                    if let Ok((major, _minor)) = device.compute_capability() {
                        major >= 7
                    } else {
                        false
                    }
                })
                .collect()
        })
    };
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Device(pub c_int);

impl Device {
    fn compute_capability(&self) -> Result<(c_int, c_int), Error> {
        let mut minor = 0;
        let mut major = 0;

        let success = unsafe { cudaDeviceGetAttribute(&mut minor, CudaDeviceAttr::ComputeCapabilityMinor, self.0) };
        if success != 0 {
            return Err(Error::CudaError(success));
        }

        let success = unsafe { cudaDeviceGetAttribute(&mut major, CudaDeviceAttr::ComputeCapabilityMajor, self.0) };
        if success != 0 {
            return Err(Error::CudaError(success));
        }

        Ok((major, minor))
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        let previous_device = Device::current()?;

        if self.0 != previous_device.0 {
            self.set_current()?;
        }

        let success = unsafe { cudaDeviceSynchronize() };

        if success != 0 {
            return Err(Error::CudaError(success));
        }

        if previous_device.0 != self.0 {
            previous_device.set_current()?
        }

        Ok(())
    }

    pub fn current() -> Result<Device, Error> {
        let mut device_id = 0;
        let success = unsafe { cudaGetDevice(&mut device_id) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(Device(device_id))
        }
    }

    pub fn is_current(&self) -> Result<bool, Error> {
        let mut device_id = 0;
        let success = unsafe { cudaGetDevice(&mut device_id) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(device_id == self.0)
        }
    }

    pub fn set_current(&self) -> Result<(), Error> {
        let success = unsafe { cudaSetDevice(self.0) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(())
        }
    }
}

/// Returns a list containing all CUDA enabled devices with sufficient compute
/// capabilities (>= 7.0).
pub fn get_all_devices() -> Result<Vec<Device>, Error> {
    DEVICES.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_current_device() {
        for device in get_all_devices().unwrap() {
            assert_eq!(device.set_current(), Ok(()));
            assert_eq!(device.synchronize(), Ok(()));
            assert_eq!(device.is_current(), Ok(true));

            assert_eq!(Device::current(), Ok(device));
        }
    }
}
