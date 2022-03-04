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

#[allow(non_camel_case_types)]
pub type cudaDataType = DataType;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DataType {
    Real16F = 2,
    Real32F = 0,
}

impl DataType {
    pub fn size_in_bytes(&self) -> usize {
        match *self {
            DataType::Real16F => 2,
            DataType::Real32F => 4
        }
    }
}
