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

use util::f16::*;

/// A typeless array that can contain either a single or half precision
/// array without a generic.
pub enum Array {
    Single(Box<[f32]>),
    Half(Box<[f16]>)
}

impl Array {
    pub fn from_f16(src: Box<[f16]>) -> Array {
        Array::Half(src)
    }

    pub fn from_f32(src: Box<[f32]>) -> Array {
        Array::Single(src)
    }

    /// Returns true if this array contains an array of `f16`.
    pub fn is_half(&self) -> bool {
        match *self {
            Array::Single(_) => false,
            Array::Half(_) => true
        }
    }

    pub fn get(&self, index: usize) -> f32 {
        match *self {
            Array::Single(ref src) => src[index],
            Array::Half(ref src) => f32::from(src[index])
        }
    }
}
