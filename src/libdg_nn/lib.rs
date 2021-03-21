// Copyright 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
#![feature(test)]
#![feature(vec_into_raw_parts)]

extern crate crossbeam_channel;
extern crate dashmap;
extern crate dg_cuda;
extern crate dg_go;
extern crate dg_utils;
extern crate libc;
extern crate memchr;
#[cfg(test)] extern crate test;

mod error;
mod graph;
mod layers;
mod loader;
mod network;
mod output_map;
mod tensor;

pub use self::error::Error;
pub use self::graph::{Workspace, forward};
pub use self::network::{Network, WorkspaceGuard};
pub use self::output_map::*;
