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

/// A typeless container that can contain either a single or half precision
/// value without a generic.
pub enum Singleton {
    Single(f32),
    Half(f16)
}

impl Singleton {
    pub fn from_f16(src: f16) -> Singleton {
        Singleton::Half(src)
    }

    pub fn from_f32(src: f32) -> Singleton {
        Singleton::Single(src)
    }

    /// Returns true if this singletons contains an `f16`.
    pub fn is_half(&self) -> bool {
        match *self {
            Singleton::Single(_) => false,
            Singleton::Half(_) => true
        }
    }

    pub fn get(&self) -> f32 {
        match *self {
            Singleton::Single(src) => src,
            Singleton::Half(src) => f32::from(src)
        }
    }
}
