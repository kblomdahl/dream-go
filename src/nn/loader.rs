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

use libc::{c_void};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::ptr;

use nn::ffi::cuda::*;
use util::b85;

fn skip_until<I>(iter: &mut I, stop: char) -> String
    where I: Iterator<Item=char>
{
    let mut out: String = String::new();

    loop {
        let ch = iter.next();

        if ch.is_none() || ch == Some(stop) {
            break
        }

        out.push(ch.unwrap());
    }

    out
}

pub fn load(path: &Path) -> Option<HashMap<String, *const c_void>> {
    if let Ok(file) = File::open(path) {
        let mut iter = BufReader::new(file).chars().map(|ch| ch.unwrap());
        let mut out: HashMap<String, *const c_void> = HashMap::new();

        // parse entries of the format -- "name": "value"
        loop {
            // skip until next quote
            skip_until(&mut iter, '"');

            // name of the tensor
            let name = skip_until(&mut iter, '"');
            if name.is_empty() {
                break
            }

            // skip until next quote
            skip_until(&mut iter, '"');            

            // value of the tensor
            let value = skip_until(&mut iter, '"');
            let tensor = b85::decode(&value).unwrap();

            for (i, element) in tensor.iter().enumerate() {
                if !element.is_finite() {
                    println!("{}: element {} is not finite -- {}", name, i, element);
                }
            }

            // copy the value of this tensor to the device
            unsafe {
                let mut w = ptr::null_mut();
                let size = 4 * tensor.len();

                assert!(cudaMalloc(&mut w, size).is_ok());
                assert!(cudaMemcpy(
                    w,
                    tensor.as_ptr() as *const c_void,
                    size,
                    MemcpyKind::HostToDevice
                ).is_ok());

                out.insert(name, w as *const c_void);
            }
        }

        Some(out)
    } else {
        None
    }
}
