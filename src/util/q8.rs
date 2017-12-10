// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

/// 16-bit floating point numbers as defined in IEEE 754-2008.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct q8(i8);

impl Default for q8 {
    fn default() -> q8 { q8(0) }
}

impl ::std::fmt::Debug for q8 {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::result::Result<(), ::std::fmt::Error> {
        write!(f, "{}", f32::from(*self))
    }
}

impl From<q8> for f32 {
    fn from(value: q8) -> f32 {
        let q8(quan) = value;

        (quan as f32) / 127.0
    }
}

impl From<f32> for q8 {
    fn from(value: f32) -> q8 {
        let quan = (127.0 * value) as i32;

        if quan > 127 {
            q8(127)
        } else if quan < -127 {
            q8(-127)
        } else {
            q8(quan as i8)
        }
    }
}
