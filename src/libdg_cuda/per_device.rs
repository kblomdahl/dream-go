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

use crate::error::{cudaError_t, Error};

use std::ops::{Deref, DerefMut};
use libc::c_int;

#[link(name = "cuda")]
#[link(name = "cudart")]
extern {
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaGetDevice(device_id: *mut c_int) -> cudaError_t;
    pub fn cudaSetDevice(device_id: c_int) -> Error;
}

pub struct PerDevice<T: Default> {
    values: Vec<T>
}

impl<T: Default> Deref for PerDevice<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        let mut device_id = 0;
        unsafe { cudaGetDevice(&mut device_id) };

        self.values.get(device_id as usize).unwrap()
    }
}

impl<T: Default> DerefMut for PerDevice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let mut device_id = 0;
        unsafe { cudaGetDevice(&mut device_id) };

        self.values.get_mut(device_id as usize).unwrap()
    }
}

impl<T: Default> PerDevice<T> {
    pub fn new() -> Result<PerDevice<T>, Error> {
        let mut count = 0;
        let error = unsafe { cudaGetDeviceCount(&mut count) };

        error.into_result(Self {
            values: (0..count).map(|device_id| {
                let mut prev_device_id = 0;
                unsafe { cudaGetDevice(&mut prev_device_id) };
                unsafe { cudaSetDevice(device_id) };
                let out = T::default();
                unsafe { cudaSetDevice(prev_device_id) };
                out
            }).collect()
        })
    }
}

#[cfg(test)]
mod tests {
    // pass
}
