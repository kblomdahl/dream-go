// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
use libc::c_int;

#[allow(non_camel_case_types)]
pub(super) type cublasStatus_t = Status;

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq)]
pub struct Status {
    value: c_int
}

impl Status {
    pub fn success() -> Self {
        Self { value: 0 }
    }

    pub fn into_result<Ok>(self, ok: Ok) -> Result<Ok, Self> {
        if self.value == 0 {
            Ok(ok)
        } else {
            Err(self)
        }
    }
}

impl Debug for Status {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_is_ok() {
        assert_eq!(Status::success().into_result(1), Ok(1));
    }
}
