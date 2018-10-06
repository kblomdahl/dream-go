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

/// Returns the number of elements in the given array that are `0`. The
/// input arrays length must be a divider of `16` (for SIMD reasons).
/// 
/// # Arguments
/// 
/// * `array` - 
/// 
#[target_feature(enable = "avx,avx2,popcnt")]
unsafe fn _count_zeros(array: &[u8]) -> usize {
    debug_assert!(array.len() == 384);

    use std::arch::x86_64::*;

    let zero = _mm256_setzero_si256();
    let mut count = 0;

    for i in 0..12 {
        let x = _mm256_loadu_si256(array.get_unchecked(32*i) as *const u8 as *const _);
        let c = _mm256_cmpeq_epi8(x, zero);
        let m = _mm256_movemask_epi8(c);  // set one bit for every match in `c`

        count += _popcnt32(m);  // count the number of matches in `c`
    }

    count as usize
}

/// Returns the number of elements in the given array that are `0`. The
/// input arrays length must be a divider of `16` (for SIMD reasons).
/// 
/// # Arguments
/// 
/// * `array` - 
/// 
#[inline(always)]
pub fn count_zeros(array: &[u8]) -> usize {
    if is_x86_feature_detected!("avx2")  {
        unsafe { _count_zeros(array) }
    } else {
        (0..361).filter(|&i| array[i] == 0).count()
    }
}

#[cfg(test)]
mod tests {
    use test::Bencher;
    use go::asm;

    #[bench]
    fn count_zeros(b: &mut Bencher) {
        let mut array = [1u8; 384];

        array[  0] = 0; array[  3] = 0; array[ 18] = 0; array[ 21] = 0;
        array[ 42] = 0; array[ 71] = 0; array[121] = 0; array[209] = 0;
        array[212] = 0; array[281] = 0; array[300] = 0; array[311] = 0;

        // check that the result is correct before the tight loop
        assert_eq!(asm::count_zeros(&array), 12);

        // benchmark our implementation
        b.iter(move || {
            asm::count_zeros(&array)
        });
    }
}
