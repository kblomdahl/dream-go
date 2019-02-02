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
unsafe fn _argmax_f32(original_array: &[f32]) -> Option<usize> {
    debug_assert_eq!(original_array.len() % 8, 0);

    let steps = original_array.len() / 8;
    let mut array = original_array.as_ptr();
    let mut so_far = _mm256_broadcast_ss(&::std::f32::NEG_INFINITY);
    let mut index: usize = 0;

    for i in 0..steps {
        let x = _mm256_loadu_ps(array as *const f32 as *const _);

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
        let eq = _mm256_cmp_ps(so_far, x, _CMP_EQ_OQ);
        let eq = _mm256_movemask_ps(eq) as u32;

        if eq != 0 {
            let trailing_zeros = _mm_tzcnt_32(eq) as usize;

            index = 8 * i + trailing_zeros;
        }

        array = array.add(8);
    }

    if (*original_array.get_unchecked(index)).is_finite() {
        Some(index)
    } else {
        None
    }
}

/// Returns the index of the maximum value in the given array. If multiple
/// indices share the same value, then which is returned is undefined.
/// 
/// # Arguments
/// 
/// * `array` -
/// 
#[inline(always)]
pub fn argmax_f32(array: &[f32]) -> Option<usize> {
    if is_x86_feature_detected!("avx2")  {
        unsafe { _argmax_f32(array) }
    } else {
        (0..362).filter(|&i| array[i].is_finite())
            .max_by_key(|&i| OrderedFloat(array[i]))
    }
}

/// Returns the index of the maximum value in the given array. If multiple
/// indices share the same value, then which is returned is undefined.
///
/// # Arguments
///
/// * `array` -
///
#[target_feature(enable = "avx,avx2,bmi1")]
unsafe fn _argmax_i32(array: &[i32]) -> Option<usize> {
    debug_assert_eq!(array.len() % 8, 0);

    let steps = array.len() / 8;
    let mut array = array.as_ptr();
    let mut so_far = _mm256_set1_epi32(::std::i32::MIN);
    let mut index: usize = ::std::usize::MAX;

    for i in 0..steps {
        let x = _mm256_loadu_si256(array as *const _);

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
        let y = _mm256_shuffle_epi32(x, 0xb1);
        let z = _mm256_max_epi32(x, y);
        let y = _mm256_shuffle_epi32(z, 0x4e);
        let z = _mm256_max_epi32(z, y);
        let y = _mm256_permute2f128_si256(z, z, 0x01);
        let z = _mm256_max_epi32(z, y);

        // determine the index of the element from our horizontal
        // maximum that is now in `so_far`.
        so_far = _mm256_max_epi32(so_far, z);
        let eq = _mm256_cmpeq_epi32(so_far, x);
        let eq = _mm256_movemask_epi8(eq) as u32;

        if eq != 0 {
            let trailing_zeros = _mm_tzcnt_32(eq) as usize;

            index = 8 * i + trailing_zeros / 4;
        }

        array = array.add(8);
    }

    if index == ::std::usize::MAX {
        None
    } else {
        Some(index)
    }
}

/// Returns the index of the maximum value in the given array. If multiple
/// indices share the same value, then which is returned is undefined.
///
/// # Arguments
///
/// * `array` -
///
#[inline(always)]
pub fn argmax_i32(array: &[i32]) -> Option<usize> {
    if is_x86_feature_detected!("avx2")  {
        unsafe { _argmax_i32(array) }
    } else {
        (0..array.len()).max_by_key(|&i| array[i])
    }
}

#[cfg(test)]
mod tests {
    use test::{self, Bencher};
    use super::*;

    #[bench]
    fn argmax_f32_each(b: &mut Bencher) {
        let mut array = [::std::f32::NEG_INFINITY; 368];

        // test setting each element within an eight lane as the maximum to
        // ensure nothing is lost
        for i in 0..362 {
            array[i] = 2.0 + (i as f32);

            assert_eq!(argmax_f32(&array), Some(i));
        }

        let array = test::black_box(array);

        b.iter(move || {
            argmax_f32(&array)
        });
    }

    #[test]
    fn check_argmax_f32() {
        let mut array = [-1.0f32; 368];
        array[234] = -0.1;

        assert_eq!(argmax_f32(&array), Some(234));
    }

    #[test]
    fn check_argmax_i32() {
        let mut array = [-2; 368];
        array[127] = -1;
        array[257] = 0;

        assert_eq!(argmax_i32(&array), Some(257));
    }

    #[test]
    fn check_all_inf() {
        let array = [::std::f32::NEG_INFINITY; 368];

        assert_eq!(argmax_f32(&array), None);
    }

    #[test]
    fn check_all_nan() {
        let mut array = [::std::f32::NEG_INFINITY; 368];
        for i in 0..362 {
            array[i] = ::std::f32::NAN;
        }

        assert_eq!(argmax_f32(&array), None);
    }
}
