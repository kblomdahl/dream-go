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

use nn::cuda::*;

const BASE_85: [char; 85] = [
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
	'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
	'U', 'V', 'W', 'X', 'Y', 'Z',
	'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
	'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
	'u', 'v', 'w', 'x', 'y', 'z',
	'!', '#', '$', '%', '&', '(', ')', '*', '+', '-',
	';', '<', '=', '>', '?', '@', '^', '_',	'`', '{',
	'|', '}', '~'
];

lazy_static! {
    /// Lookup table from alphabet characters to its bit value
    static ref DECODE_85: [i8; 256] = {
        let mut out = [-1; 256];

        for (i, b) in BASE_85.iter().enumerate() {
            out[*b as usize] = i as i8;
        }

        out
    };
}

/// Decode a RFC 1924 (Ascii85) encoded string of FP32 values and returns
/// an array of the FP32 numbers it represents.
fn decode_b85(input: &str) -> Option<Box<[f32]>> {
    let mut output = vec! [];
    let mut iter = input.chars();

    'outer: loop {
        // decode the alphabet into raw bits
        let mut acc: u32 = 0;

        for _ in 0..5 {
            if let Some(ch) = iter.next() {
                let de = unsafe { *DECODE_85.get_unchecked(ch as usize) };
                if de < 0 {
                    return None;  // invalid character
                }

                acc = 85 * acc + de as u32;
            } else {
                break 'outer;
            }
        }

        // encode the bits into 16-bit floating point numbers
        let mut dst = [0; 4];

        for i in 0..4 {
            dst[i] = acc as u8;
            acc >>= 8;
        }

        output.push(f32::from_bits(
            ((dst[0] as u32) << 24)
            | ((dst[1] as u32) << 16)
            | ((dst[2] as u32) << 8)
            | (dst[3] as u32))
        );
    }

    Some(output.into_boxed_slice())
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
            let tensor = decode_b85(&value).unwrap();

            for (i, element) in tensor.iter().enumerate() {
                if !element.is_finite() {
                    println!("{}: element {} is not finite -- {}", name, i, element);
                }
            }

            // copy the value of this tensor to the device
            unsafe {
                let mut w = ptr::null_mut();
                let size = 4 * tensor.len();

                assert_eq!(cudaMalloc(&mut w, size), Error::Success);
                assert_eq!(cudaMemcpy(
                    w,
                    tensor.as_ptr() as *const c_void,
                    size,
                    MemcpyKind::HostToDevice
                ), Error::Success);

                out.insert(name, w as *const c_void);
            }
        }

        Some(out)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use nn::loader::decode_b85;

    #[test]
    fn pi_e() {
        let string = "+Yd=VRQN4G";

        assert_eq!(
            decode_b85(string),
            Some(vec! [3.14159265, 2.71828183].into_boxed_slice())
        );
    }

    // Test that we can handle padding correctly
    #[test]
    fn _1234567() {
        let string = "004kL0000$002Nh004kM005vs006*1007`X";

        assert_eq!(
            decode_b85(string),
            Some(vec! [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].into_boxed_slice())
        );
    }
}
