// Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
extern crate go;
extern crate test;

use go::util::extract_example::*;

use std::ffi::CString;
use std::fs::File;
use std::io::{BufRead, BufReader};
use test::Bencher;

#[bench]
fn all_succeed(b: &mut Bencher) {
    let f = File::open("fixtures/foxwq.sgf").unwrap();
    let lines = BufReader::new(&f).lines()
        .map(|x| CString::new(x.unwrap()).unwrap())
        .collect::<Vec<_>>();
    let mut example = Example::default();

    b.iter(move || {
        for line in &lines {
            let code = unsafe {
                extract_single_example(line.as_ptr(), &mut example)
            };

            assert_eq!(code, 0, "Code {}: {:?}", code, line);
        }
    });
}
