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

use crate::Err;

use dg_cuda as cuda;
use dg_utils::types::f16;

#[derive(Debug, PartialEq)]
enum Value {
    Float(Vec<f32>),
    Half(Vec<f16>),
    Integer(Vec<i32>),
}

#[derive(Debug, PartialEq)]
pub struct Variable {
    value: Value
}

impl From<Vec<f32>> for Variable {
    fn from(v: Vec<f32>) -> Self {
        Self { value: Value::Float(v) }
    }
}

impl From<Vec<f16>> for Variable {
    fn from(v: Vec<f16>) -> Self {
        Self { value: Value::Half(v) }
    }
}

impl From<Vec<i32>> for Variable {
    fn from(v: Vec<i32>) -> Self {
        Self { value: Value::Integer(v) }
    }
}

pub trait AsSlice<T: Sized> {
    fn as_slice(&self) -> Result<&[T], Err>;
}

impl Variable {
    pub fn as_ptr(&self, stream: &cuda::Stream) -> Result<cuda::Ptr, cuda::Error> {
        match &self.value {
            Value::Float(v) => cuda::Ptr::from_slice(&v, stream),
            Value::Half(v) => cuda::Ptr::from_slice(&v, stream),
            Value::Integer(v) => cuda::Ptr::from_slice(&v, stream)
        }
    }
}

impl AsSlice<f32> for Variable {
    fn as_slice(&self) -> Result<&[f32], Err> {
        match &self.value {
            Value::Float(v) => Ok(v),
            _ => Err(Err::UnexpectedValue)
        }
    }
}

impl AsSlice<f16> for Variable {
    fn as_slice(&self) -> Result<&[f16], Err> {
        match &self.value {
            Value::Half(v) => Ok(v),
            _ => Err(Err::UnexpectedValue)
        }
    }
}

impl AsSlice<i32> for Variable {
    fn as_slice(&self) -> Result<&[i32], Err> {
        match &self.value {
            Value::Integer(v) => Ok(v),
            _ => Err(Err::UnexpectedValue)
        }
    }
}
