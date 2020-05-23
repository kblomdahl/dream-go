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

use std::fmt::{self, Debug, Formatter};
use std::ffi::CStr;
use libc::c_char;

#[allow(non_camel_case_types)]
pub(super) type cudaError_t = Error;

#[link(name = "cudnn")]
extern {
    fn cudaGetErrorName(error: cudaError_t) -> *const c_char;
}

#[repr(i32)]
#[derive(Clone, Copy, PartialEq)]
pub enum Error {
    Success = 0,
}

impl Error {
    pub fn into_result<Ok>(self, ok: Ok) -> Result<Ok, Self> {
        if self == Error::Success {
            Ok(ok)
        } else {
            Err(self)
        }
    }
}

impl Debug for Error {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        let ptr = unsafe { cudaGetErrorName(*self) };
        let c_str = unsafe { CStr::from_ptr(ptr) };

        match c_str.to_str() {
            Ok(s) => write!(f, "{}", s),
            _ => write!(f, "Illegal error code"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::transmute;
    use super::*;

    #[test]
    fn success_is_ok() {
        assert_eq!(Error::Success.into_result(1), Ok(1));
    }

    #[test]
    fn invalid_value() {
        let status: Error = unsafe { transmute(1) };

        assert_eq!(format!("{:?}", status), "cudaErrorInvalidValue");
    }
}
