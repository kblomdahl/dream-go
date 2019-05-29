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

extern crate dg_go;

use dg_go::utils::extract_example::*;

use std::ffi::CString;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[test]
fn all_succeed() {
    let f = File::open("fixtures/example_games.sgf").unwrap();
    let mut example = Example::default();

    for (line_nr, line) in BufReader::new(&f).lines().enumerate() {
        if let Ok(line) = line {
            let c_string = CString::new(line).unwrap();
            let code = unsafe {
                extract_single_example(c_string.as_ptr(), &mut example)
            };

            assert_eq!(code, 0, "Line {}, Code {}: {:?}", line_nr, code, c_string);
        } else {
            panic!();
        }
    }
}
