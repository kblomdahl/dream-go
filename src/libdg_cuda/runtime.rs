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

use super::Error;

use libc::c_int;

#[link(name = "cuda")]
#[link(name = "cudart")]
extern {
    fn cudaRuntimeGetVersion(version: *mut c_int) -> c_int;
}

pub fn runtime_version() -> Result<usize, Error> {
    let mut version = 0;
    let success = unsafe { cudaRuntimeGetVersion(&mut version) };

    if success != 0 {
        Err(Error::CudaError(success))
    } else {
        Ok(version as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_runtime_version() {
        assert!(runtime_version().is_ok());
    }
}
