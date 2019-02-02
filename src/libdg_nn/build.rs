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

#[link(name = "cudnn")]
extern {
    fn cudnnGetProperty(property: i32, value: *mut i32) -> i32;
}

fn main() {
    // determine the cuDNN version
    let mut major_version = 0;
    let mut minor_version = 0;
    let mut patch_level = 0;

    unsafe {
        cudnnGetProperty(0, &mut major_version as *mut i32);
        cudnnGetProperty(1, &mut minor_version as *mut i32);
        cudnnGetProperty(2, &mut patch_level as *mut i32);
    }

    // if the cuDNN version is at least 7.0.1 then enable the `tensor-core` feature
    if major_version >= 7 && (major_version != 7 || minor_version >= 1 || patch_level >= 1) {
        println!("cargo:rustc-cfg=feature=\"tensor-core\"");
    }
}
