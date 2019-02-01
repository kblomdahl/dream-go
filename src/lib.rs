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
#![feature(core_intrinsics, link_llvm_intrinsics, manually_drop_take)]
#![feature(test)]

extern crate cpu_time;
extern crate crossbeam_channel;
extern crate crossbeam_utils;
extern crate go;
#[macro_use] extern crate lazy_static;
extern crate libc;
extern crate memchr;
extern crate ordered_float;
extern crate rand;
extern crate regex;
#[cfg(test)] extern crate test;
extern crate time;

pub mod gtp;
pub mod mcts;
pub mod nn;
pub mod parallel;
pub mod util;
