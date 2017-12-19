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

/// 16-bit floating point numbers as defined in IEEE 754-2008.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct f16(u16);

impl f16 {
    /// Wrap the given bits as an half precision floating point number.
    /// 
    /// # Arguments
    /// 
    /// * `bits` - the bits to wrap
    /// 
    pub fn from_bits(bits: u16) -> f16 {
        f16(bits)
    }

    /// Returns the wrapped bits.
    pub fn to_bits(&self) -> u16 {
        let f16(bits) = *self;

        bits
    }
}

impl Default for f16 {
    fn default() -> f16 { f16(0) }
}

impl ::std::fmt::Debug for f16 {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::result::Result<(), ::std::fmt::Error> {
        write!(f, "{}", f32::from(*self))
    }
}

extern "C" {
    #[link_name = "llvm.convert.to.fp16.f32"]
    fn convert_to_fp16_f32(f: f32) -> u16;

    #[link_name = "llvm.convert.from.fp16.f32"]
    fn convert_from_fp16_f32(f: u16) -> f32;
}

impl From<f16> for f32 {
    fn from(value: f16) -> f32 {
        let f16(bits) = value;

        unsafe { convert_from_fp16_f32(bits) }
    }
}

impl From<f32> for f16 {
    fn from(value: f32) -> f16 {
        f16(unsafe { convert_to_fp16_f32(value) })
    }
}

#[cfg(test)]
mod tests {
    use test::{self, Bencher};
    use util::f16::*;

    #[test]
    fn from_f16_to_f32() {
        assert_eq!(f32::from(f16::from_bits(0x4170)), 2.71875);  // e
        assert_eq!(f32::from(f16::from_bits(0x4248)), 3.140625);  // pi
        assert_eq!(f32::from(f16::from_bits(0x3518)), 0.31835938);  // 1/pi
        assert_eq!(f32::from(f16::from_bits(0x398c)), 0.6933594);  // ln 2
        assert_eq!(f32::from(f16::from_bits(0x36f3)), 0.43432617);  // log10 e
        assert_eq!(f32::from(f16::from_bits(0x3dc5)), 1.4423828);  // log2 e
        assert_eq!(f32::from(f16::from_bits(0x3da8)), 1.4140625);  // sqrt 2
    }

    #[test]
    fn from_f32_to_f16() {
        assert_eq!(f16::from(::std::f32::consts::PI).to_bits(), 0x4248);  // pi
        assert_eq!(f16::from(::std::f32::consts::E).to_bits(), 0x4170);  // e
    }

    #[bench]
    fn convert_to_fp16_f32(b: &mut Bencher) {
        b.iter(|| {
            let pi = test::black_box(3.14f32);

            f16::from(pi)
        })
    }

    #[bench]
    fn convert_from_fp16_f32(b: &mut Bencher) {
        let e = f16::from_bits(0x4248);

        b.iter(|| {
            let e = test::black_box(e);

            f32::from(e)
        })
    }
}
