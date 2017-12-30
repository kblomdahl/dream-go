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
use nn::ffi::cudnn::*;
use util::b85;
use util::f16::*;

enum Tensor {
    Float(Box<[f32]>),
    Half(Box<[f16]>)
}

impl Tensor {
    unsafe fn as_ptr(&self) -> *const c_void {
        match *self {
            Tensor::Float(ref b) => b.as_ptr() as *const c_void,
            Tensor::Half(ref b) => b.as_ptr() as *const c_void
        }
    }

    fn size_in_bytes(&self) -> usize {
        match *self {
            Tensor::Float(ref b) => 4 * b.len(),
            Tensor::Half(ref b) => 2 * b.len()
        }
    }

    #[cfg(debug_assertions)]
    fn as_vec(&self) -> Vec<f32> {
        match *self {
            Tensor::Float(ref b) => b.iter().map(|&v| v).collect(),
            Tensor::Half(ref b) => b.iter().map(|&v| f32::from(v)).collect()
        }
    }
}

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

pub fn load(path: &Path, data_type: DataType) -> Option<HashMap<String, *const c_void>> {
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
            let tensor = {
                if data_type == DataType::Float {
                    Tensor::Float(b85::decode::<f32>(&value).unwrap())
                } else {
                    assert_eq!(data_type, DataType::Half);

                    Tensor::Half(b85::decode::<f16>(&value).unwrap())
                }
            };

            #[cfg(debug_assertions)]
            for (i, element) in tensor.as_vec().iter().enumerate() {
                if !element.is_finite() {
                    eprintln!("{}: element {} is not finite -- {}", name, i, element);
                }
            }

            // copy the value of this tensor to the device
            unsafe {
                let mut w = ptr::null_mut();
                let size = tensor.size_in_bytes();

                assert!(cudaMalloc(&mut w, size).is_ok());
                assert!(cudaMemcpy(
                    w,
                    tensor.as_ptr(),
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
