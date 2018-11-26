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

use std::arch::x86_64::*;

/// Returns the sum of all finite elements in `array`.
///
/// # Arguments
///
/// * `array` -
///
#[target_feature(enable = "avx,avx2")]
unsafe fn _sum_finite_f32(array: &[f32]) -> f32 {
    debug_assert_eq!(array.len() % 8, 0);

    let sign_mask =_mm256_set1_ps(-0.0);
    let inf = _mm256_set1_ps(::std::f32::INFINITY);

    let steps = array.len() / 8;
    let mut array = array.as_ptr();
    let mut so_far = _mm256_setzero_ps();

    for _i in 0..steps {
        let x = _mm256_loadu_ps(array as *const _);

        // check if x has any infinity
        let is_inf = _mm256_andnot_ps(sign_mask, x);
        let is_inf = _mm256_cmp_ps(is_inf, inf, _CMP_EQ_OQ);

        // check if x has any NaN
        let is_not_nan = _mm256_cmp_ps(x, x, _CMP_EQ_OQ);

        // sum
        so_far = _mm256_add_ps(
            so_far,
            _mm256_andnot_ps(is_inf, _mm256_and_ps(is_not_nan, x))
        );

        array = array.add(8);
    }

    // horizontal sum (this is faster than a fancy `_mm256_hadd_ps` dance
    let mut out: [f32; 8] = ::std::mem::uninitialized();
    _mm256_store_ps(out.as_mut_ptr() as *mut _, so_far);

    ((out[0] + out[1]) + (out[2] + out[3])) + ((out[4] + out[5]) + (out[6] + out[7]))
}

/// Returns the sum of all finite elements in `array`.
///
/// # Arguments
///
/// * `array` -
///
#[inline(always)]
pub fn sum_finite_f32(array: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2")  {
        unsafe { _sum_finite_f32(array) }
    } else {
        array.iter().filter(|x| x.is_finite()).sum::<f32>()
    }
}

/// Returns the sum of all elements in `array`.
///
/// # Arguments
///
/// * `array` -
///
#[target_feature(enable = "avx,avx2")]
unsafe fn _sum_i32(array: &[i32]) -> i32 {
    debug_assert_eq!(array.len() % 8, 0);

    let steps = array.len() / 8;
    let mut array = array.as_ptr();
    let mut so_far = _mm256_setzero_si256();

    for _i in 0..steps {
        let x = _mm256_loadu_si256(array as *const _);

        so_far = _mm256_add_epi32(so_far, x);
        array = array.add(8);
    }

    // horizontal sum (this is faster than a fancy `_mm256_hadd_epi32` dance
    let mut out: [i32; 8] = ::std::mem::uninitialized();
    _mm256_store_si256(out.as_mut_ptr() as *mut _, so_far);

    ((out[0] + out[1]) + (out[2] + out[3])) + ((out[4] + out[5]) + (out[6] + out[7]))
}

/// Returns the sum of all elements in `array`.
///
/// # Arguments
///
/// * `array` -
///
#[inline(always)]
pub fn sum_i32(array: &[i32]) -> i32 {
    if is_x86_feature_detected!("avx2")  {
        unsafe { _sum_i32(array) }
    } else {
        array.iter().sum::<i32>()
    }
}

#[cfg(test)]
mod tests {
    use test::{self, Bencher};
    use super::*;

    #[bench]
    fn sum(b: &mut Bencher) {
        let mut array = [::std::f32::NEG_INFINITY; 368];

        // test setting each element within an eigth lane as the maximum to
        // ensure nothing is lost
        for i in 0..200 {
            array[i] = 2.0 + (i as f32);
        }

        let array = test::black_box(array);

        b.iter(move || {
            sum_finite_f32(&array)
        });
    }

    #[test]
    fn check_normal() {
        let mut array = [0.0f32; 368];

        for i in 0..8 {
            array[i] = -(i as f32);
        }

        assert_eq!(sum_finite_f32(&array), -28.0);
    }

    #[test]
    fn check_nan() {
        let mut array = [1.0f32; 368];
        array[0] = ::std::f32::NAN;

        assert_eq!(sum_finite_f32(&array), 367.0);
    }

    #[test]
    fn check_inf() {
        let mut array = [1.0f32; 368];
        array[0] = ::std::f32::INFINITY;

        assert_eq!(sum_finite_f32(&array), 367.0);
    }

    #[test]
    fn check_neg_inf() {
        let mut array = [1.0f32; 368];
        array[0] = ::std::f32::NEG_INFINITY;

        assert_eq!(sum_finite_f32(&array), 367.0);
    }
}
