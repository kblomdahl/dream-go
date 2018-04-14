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
    Single(Vec<f32>),
    Half(Vec<f16>),
    Int8(Vec<q8>),
    None,
}

impl Array {
    /// Returns an empty, untyped array
    pub fn empty() -> Array {
        Array::None
    }

    /// Returns true if this array contains an array of `f16`.
    pub fn is_half(&self) -> bool {
        match *self {
            Array::Half(_) => true,
            _ => false
        }
    }

    pub fn split_off(&mut self, index: usize) -> Array {
        match *self {
            Array::Single(ref mut src) => Array::from(src.split_off(index)),
            Array::Half(ref mut src) => Array::from(src.split_off(index)),
            Array::Int8(ref mut src) => Array::from(src.split_off(index)),
            Array::None => unreachable!()
        }
    }

    pub fn extend_from_slice(&mut self, other: Array) {
        match (self, other) {
            (x @ Array::None, other) => {
                ::std::mem::replace(x, other);
            },
            (Array::Single(ref mut src), Array::Single(ref other)) => src.extend_from_slice(&other),
            (Array::Half(ref mut src), Array::Half(ref other)) => src.extend_from_slice(&other),
            (Array::Int8(ref mut src), Array::Int8(ref other)) => src.extend_from_slice(&other),
            _ => unreachable!()
        }
    }

    pub fn get(&self, index: usize) -> f32 {
        match *self {
            Array::Single(ref src) => src[index],
            Array::Half(ref src) => f32::from(src[index]),
            Array::Int8(ref src) => f32::from(src[index]),
            Array::None => unreachable!()
        }
    }

    pub fn len(&self) -> usize {
        match *self {
            Array::Single(ref src) => src.len(),
            Array::Half(ref src) => src.len(),
            Array::Int8(ref src) => src.len(),
            Array::None => unreachable!()
        }
    }
}

impl From<Vec<f32>> for Array {
    fn from(other: Vec<f32>) -> Array {
        Array::Single(other)
    }
}

impl From<Vec<f16>> for Array {
    fn from(other: Vec<f16>) -> Array {
        Array::Half(other)
    }
}

impl From<Vec<q8>> for Array {
    fn from(other: Vec<q8>) -> Array {
        Array::Int8(other)
    }
}

impl From<Array> for Vec<f32> {
    fn from(other: Array) -> Vec<f32> {
        match other {
            Array::Single(b) => b,
            _ => unreachable!()
        }
    }
}

impl From<Array> for Vec<f16> {
    fn from(other: Array) -> Vec<f16> {
        match other {
            Array::Half(b) => b,
            _ => unreachable!()
        }
    }
}

impl From<Array> for Vec<q8> {
    fn from(other: Array) -> Vec<q8> {
        match other {
            Array::Int8(b) => b,
            _ => unreachable!()
        }
    }
}
