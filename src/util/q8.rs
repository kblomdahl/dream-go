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

pub const RANGE: f32 = 1.0;

/// 8-bit quantized number in the range [-RANGE, RANGE].
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct q8(i8);

impl q8 {
    /// Returns an unscaled (not divided by `RANGE`) quantized value. This
    /// is useful if you want to do your own scaling.
    /// 
    /// # Arguments
    /// 
    /// * `value` - 
    /// 
    pub fn unscaled(value: f32) -> q8 {
        if value > RANGE {
            q8(127)
        } else if value < -RANGE {
            q8(-127)
        } else {
            q8((127.0 * value) as i8)
        }
    }
}

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

        RANGE * (quan as f32) / 127.0
    }
}

impl From<f32> for q8 {
    fn from(value: f32) -> q8 {
        if value > RANGE {
            q8(127)
        } else if value < -RANGE {
            q8(-127)
        } else {
            q8((127.0 * value / RANGE) as i8)
        }
    }
}
