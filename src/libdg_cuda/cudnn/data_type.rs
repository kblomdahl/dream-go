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

#[allow(non_camel_case_types)]
pub type cudnnDataType_t = DataType;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DataType {
    Float = 0,
    Double = 1,
    Half = 2,
    Int8 = 3,
    Int32 = 4,
    Int8x4 = 5,
    Uint8 = 6,
    Uint8x4 = 7,
    Int8x32 = 8,
}

impl DataType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DataType::Float => 4,
            DataType::Double => 8,
            DataType::Half => 2,
            DataType::Int8 => 1,
            DataType::Int32 => 4,
            DataType::Int8x4 => 1,
            DataType::Uint8 => 1,
            DataType::Uint8x4 => 1,
            DataType::Int8x32 => 1
        }
    }
}

impl Default for DataType {
    fn default() -> Self {
        DataType::Float
    }
}
