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
pub(super) type cudnnStatus_t = Status;

#[link(name = "cudnn_ops_infer")]
extern {
    fn cudnnGetErrorString(status: cudnnStatus_t) -> *const c_char;
}

#[repr(i32)]
#[derive(Clone, Copy, PartialEq)]
pub enum Status {
    Success = 0,
}

impl Status {
    pub fn into_result<Ok>(self, ok: Ok) -> Result<Ok, Self> {
        if self == Status::Success {
            Ok(ok)
        } else {
            Err(self)
        }
    }
}

impl Debug for Status {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        let ptr = unsafe { cudnnGetErrorString(*self) };
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
        assert_eq!(Status::Success.into_result(1), Ok(1));
    }

    #[test]
    fn status_not_initialized() {
        let status: Status = unsafe { transmute(1) };

        assert_eq!(format!("{:?}", status), "CUDNN_STATUS_NOT_INITIALIZED");
    }
}
