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

/// A typeless container that can contain either a single or half precision
/// value without a generic.
pub enum Singleton {
    Single(f32),
    Half(f16)
}

impl Singleton {
    pub fn get(&self) -> f32 {
        match *self {
            Singleton::Single(src) => src,
            Singleton::Half(src) => f32::from(src)
        }
    }
}

impl From<f32> for Singleton {
    fn from(other: f32) -> Singleton {
        Singleton::Single(other)
    }
}

impl From<f16> for Singleton {
    fn from(other: f16) -> Singleton {
        Singleton::Half(other)
    }
}
