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

use crate::cudnn::*;

use libc::c_int;

#[link(name = "cudnn")]
extern {
    fn cudnnGetProperty(property: c_int, value: *mut c_int) -> cudnnStatus_t;
}

fn get_property(property: c_int) -> Result<c_int, Status> {
    let mut value = 0;
    let status = unsafe { cudnnGetProperty(property, &mut value) };

    status.into_result(value)
}

pub fn get_version() -> Result<(i32, i32, i32), Status> {
    let major_version = get_property(0);
    let minor_version = get_property(1);
    let patch_level = get_property(2);

    major_version.and_then(|major_version| {
        minor_version.and_then(|minor_version| {
            patch_level.map(|patch_level| {
                (major_version, minor_version, patch_level)
            })
        })
    })
}

pub fn supports_tensor_cores() -> Result<bool, Status> {
    get_version().map(|(major_version, minor_version, patch_level)| {
        major_version >= 7 && (major_version != 7 || minor_version >= 1 || patch_level >= 1)
    })
}
