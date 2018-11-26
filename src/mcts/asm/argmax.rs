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

use ordered_float::OrderedFloat;
use std::arch::x86_64::*;

/// Returns the index of the maximum value in the given array. If multiple
/// indices share the same value, then which is returned is undefined.
/// 
/// # Arguments
/// 
/// * `array` -
/// 
#[target_feature(enable = "avx,avx2,bmi1")]
unsafe fn _argmax(array: &[f32]) -> Option<usize> {
    let mut so_far = _mm256_broadcast_ss(&::std::f32::NEG_INFINITY);
    let mut index: usize = 0;

    for i in 0..46 {
        let x = _mm256_loadu_ps(array.get_unchecked(8*i) as *const f32 as *const _);

        // this is a tree reduction of the horizontal maximum of `ymm0`
        // by shuffling the elements around and taking the maximum
        // again. For example:
        //
        // a b c d | e f g h  ymm0
        // b a d c | f e h g  ymm1 = shuffle(ymm0, [1, 0, 3, 2])
        // -----------------  ymm0 = max(ymm0, ymm1)
        // a a c c | e e g g  ymm0
        // c c a a | g g e e  ymm1 = shuffle(ymm0, [2, 3, 0, 1])
        // -----------------  ymm0 = max(ymm0, ymm1)
        // a a a a | e e e e  ymm0
        // e e e e | a a a a  ymm1 = shuffle_hilo(ymm0)
        // -----------------  ymm0 = max(ymm0, ymm1)
        // a a a a | a a a a  ymm0
        //
        let y = _mm256_permute_ps(x, 0xb1);
        let z = _mm256_max_ps(x, y);
        let y = _mm256_permute_ps(z, 0x4e);
        let z = _mm256_max_ps(z, y);
        let y = _mm256_permute2f128_ps(z, z, 0x01);
        let z = _mm256_max_ps(z, y);

        // determine the index of the element from our horizontal
        // maximum that is now in `so_far`.
        so_far = _mm256_max_ps(so_far, z);
        let eq = _mm256_cmp_ps(so_far, x, _CMP_EQ_UQ);
        let eq = _mm256_movemask_ps(eq) as u32;

        if eq != 0 {
            let trailing_zeros = _mm_tzcnt_32(eq) as usize;

            index = 8 * i + trailing_zeros;
        }
    }

    Some(index)
}

/// Returns the index of the maximum value in the given array. If multiple
/// indices share the same value, then which is returned is undefined.
/// 
/// # Arguments
/// 
/// * `array` -
/// 
#[inline(always)]
pub fn argmax(array: &[f32]) -> Option<usize> {
    if is_x86_feature_detected!("avx2")  {
        unsafe { _argmax(array) }
    } else {
        (0..362).filter(|&i| array[i].is_finite())
            .max_by_key(|&i| OrderedFloat(array[i]))
    }
}

#[cfg(test)]
mod tests {
    use test::{self, Bencher};
    use super::*;

    #[bench]
    fn argmax_each(b: &mut Bencher) {
        let mut array = [::std::f32::NEG_INFINITY; 368];

        // test setting each element within an eight lane as the maximum to
        // ensure nothing is lost
        for i in 0..362 {
            array[i] = 2.0 + (i as f32);

            assert_eq!(argmax(&array), Some(i));
        }

        let array = test::black_box(array);

        b.iter(move || {
            argmax(&array)
        });
    }

    #[test]
    fn argmax_neg() {
        let mut array = [-1.0f32; 368];
        array[234] = -0.1;

        assert_eq!(argmax(&array), Some(234));
    }
}
