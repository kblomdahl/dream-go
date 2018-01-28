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

use util::types::*;

/// A typeless array that can contain either a single or half precision
/// array without a generic.
pub enum Array {
    Single(Box<[f32]>),
    Half(Box<[f16]>),
    Int8(Box<[q8]>)
}

impl Array {
    /// Returns true if this array contains an array of `f16`.
    pub fn is_half(&self) -> bool {
        match *self {
            Array::Half(_) => true,
            _ => false
        }
    }

    pub fn get(&self, index: usize) -> f32 {
        match *self {
            Array::Single(ref src) => src[index],
            Array::Half(ref src) => f32::from(src[index]),
            Array::Int8(ref src) => f32::from(src[index])
        }
    }
}

impl From<Box<[f32]>> for Array {
    fn from(other: Box<[f32]>) -> Array {
        Array::Single(other)
    }
}

impl From<Box<[f16]>> for Array {
    fn from(other: Box<[f16]>) -> Array {
        Array::Half(other)
    }
}

impl From<Box<[q8]>> for Array {
    fn from(other: Box<[q8]>) -> Array {
        Array::Int8(other)
    }
}

impl From<Array> for Box<[f32]> {
    fn from(other: Array) -> Box<[f32]> {
        match other {
            Array::Single(b) => b,
            _ => unreachable!()
        }
    }
}

impl From<Array> for Box<[f16]> {
    fn from(other: Array) -> Box<[f16]> {
        match other {
            Array::Half(b) => b,
            _ => unreachable!()
        }
    }
}

impl From<Array> for Box<[q8]> {
    fn from(other: Array) -> Box<[q8]> {
        match other {
            Array::Int8(b) => b,
            _ => unreachable!()
        }
    }
}
