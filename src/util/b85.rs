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

const ENCODE_85: [char; 85] = [
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

        for (i, b) in ENCODE_85.iter().enumerate() {
            out[*b as usize] = i as i8;
        }

        out
    };
}

pub trait FromB85<T> {
    fn decode(bits: [u8; 4], output: &mut Vec<T>);
}

impl FromB85<u8> for u8 {
    fn decode(bits: [u8; 4], output: &mut Vec<u8>) {
        output.push(bits[3]);
        output.push(bits[2]);
        output.push(bits[1]);
        output.push(bits[0]);
    }
}

impl FromB85<i8> for i8 {
    fn decode(bits: [u8; 4], output: &mut Vec<i8>) {
        output.push(unsafe { ::std::mem::transmute::<_, _>(bits[3]) });
        output.push(unsafe { ::std::mem::transmute::<_, _>(bits[2]) });
        output.push(unsafe { ::std::mem::transmute::<_, _>(bits[1]) });
        output.push(unsafe { ::std::mem::transmute::<_, _>(bits[0]) });
    }
}

impl FromB85<f16> for f16 {
    fn decode(bits: [u8; 4], output: &mut Vec<f16>) {
        output.push(f16::from_bits(u16::from_be(((bits[3] as u16) << 8) | (bits[2] as u16))));
        output.push(f16::from_bits(u16::from_be(((bits[1] as u16) << 8) | (bits[0] as u16))));
    }
}

impl FromB85<f32> for f32 {
    fn decode(bits: [u8; 4], output: &mut Vec<f32>) {
        let raw_bits = (bits[0] as u32)
            | ((bits[1] as u32) << 8)
            | ((bits[2] as u32) << 16)
            | ((bits[3] as u32) << 24);

        output.push(f32::from_bits(u32::from_be(raw_bits)));
    }
}

/// Decode a RFC 1924 (Ascii85) encoded string of values and returns
/// an array of the numbers it represents.
/// 
/// # Arguments
/// 
/// * `input` -
/// 
pub fn decode<T, O>(input: &str) -> Option<Vec<O>>
    where T: FromB85<O> + Clone
{
    let decode_table = &*DECODE_85;  // de-ref once

    let mut output = Vec::with_capacity(input.len());
    let mut iter = input.chars();

    'outer: loop {
        // decode the alphabet into raw bits
        let mut acc: u32 = 0;

        for _ in 0..5 {
            if let Some(ch) = iter.next() {
                let de = unsafe { *decode_table.get_unchecked(ch as usize) };
                if de < 0 {
                    return None;  // invalid character
                }

                acc = 85 * acc + de as u32;
            } else {
                break 'outer;
            }
        }

        // transform the `u32` into bytes
        let dst: [u8; 4] = unsafe { ::std::mem::transmute(acc) };

        T::decode(dst, &mut output);
    }

    Some(output)
}

/// Encode an array of FP32 values as an array of FP16 values, encoded
/// according to RFC 1924 (Ascii85).
/// 
/// # Arguments
/// 
/// * `input` -
/// 
pub fn encode(input: &[f32]) -> String {
    debug_assert!(input.len() % 2 == 0);

    let mut output = String::new();
    let mut iter = input.iter();

    loop {
        if let Some(a) = iter.next() {
            let b = iter.next().unwrap();

            // cast the values to half precision before we do anything else
            let a = f16::from(*a).to_bits().to_be();
            let b = f16::from(*b).to_bits().to_be();

            // 
            let acc = ((a as usize) << 16) | (b as usize);

            output.push(ENCODE_85[acc / 52200625]);
            output.push(ENCODE_85[(acc / 614125) % 85]);
            output.push(ENCODE_85[(acc / 7225) % 85]);
            output.push(ENCODE_85[(acc / 85) % 85]);
            output.push(ENCODE_85[acc % 85]);
        } else {
            break
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use util::b85::*;

    #[test]
    fn pi_e() {
        let string = "NJ4Ny";

        assert_eq!(
            decode::<f16, _>(string),
            Some(vec! [f16::from(3.140625), f16::from(2.71875)])
        );
    }

    // Test that we can handle padding correctly
    #[test]
    fn _1234567() {
        let string = "06YLd073vn07U>s07n1-";

        assert_eq!(
            decode::<f16, _>(string).map(|it| it.into_iter().map(|value| f32::from(value)).collect()),
            Some(vec! [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0])
        );
    }

    #[test]
    fn _f32() {
        let string = "000<4";

        assert_eq!(decode::<f32, _>(string), Some(vec! [9.5]));
    }

    #[test]
    fn _encode() {
        assert_eq!(encode(&vec! [2.7578125, 3.67382812]), "gh5$D");
    }

    #[test]
    fn decode_encode() {
        let examples = vec! [
            vec! [3.140625, 2.71875],
            vec! [5.3203125, 9.9765625, 3.28320312, 8.15625, 7.109375, 1.81640625, 1.69921875, 4.4296875],
            vec! [6.37890625, 9.6171875, 2.2890625, 9.4609375, 7.8984375, 9.3125, 4.10546875, 9.390625]
        ];

        for example in examples.into_iter() {
            let result = example.iter()
                .map(|&value| f16::from(value))
                .collect::<Vec<f16>>();

            assert_eq!(decode::<f16, _>(&encode(&example)).unwrap(), result);
        }
    }
}
