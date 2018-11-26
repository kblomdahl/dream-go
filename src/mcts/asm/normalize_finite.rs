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
#[target_feature(enable = "sse2,avx,avx2")]
unsafe fn _normalize_finite_f32(array: &mut [f32], total_sum: f32) {
    debug_assert_eq!(array.len() % 4, 0);

    let total_sum = _mm_set1_ps(total_sum);
    let recip_sum = _mm_rcp_ps(total_sum);  // this is just as fast as `_mm_rcp_ss` into a broadcast

    let steps = array.len() / 4;
    let mut array = array.as_mut_ptr();

    for _i in 0..steps {
        let x = _mm_loadu_ps(array as *const _);
        let y = _mm_mul_ps(x, recip_sum);

        _mm_storeu_ps(array as *mut _, y);
        array = array.add(4);
    }
}

/// Normalize all elements in the given array, so that all finite elements
/// sum to one.
///
/// # Arguments
///
/// * `array` -
/// * `total_sum` -
///
#[inline(always)]
pub fn normalize_finite_f32(array: &mut [f32], total_sum: f32) {
    assert_ne!(total_sum, 0.0);

    if is_x86_feature_detected!("avx2")  {
        unsafe { _normalize_finite_f32(array, total_sum) }
    } else {
        let recip = total_sum.recip();

        for element in array.iter_mut() {
            *element *= recip;
        }
    }
}

#[cfg(test)]
mod tests {
    use test::{self, Bencher};
    use super::*;
    use mcts::asm::sum_finite_f32;

    #[bench]
    fn normalize(b: &mut Bencher) {
        let mut array = [::std::f32::NEG_INFINITY; 368];

        // test setting each element within an eigth lane as the maximum to
        // ensure nothing is lost
        for i in 0..362 {
            array[i] = 2.0 + (i as f32);
        }

        let total_sum = sum_finite_f32(&array);
        let array = test::black_box(array);

        b.iter(move || {
            let mut other = array.clone();

            normalize_finite_f32(&mut other, total_sum);
            other
        });
    }

    #[test]
    fn check_normal() {
        let mut vector = vec! [1.0; 64];
        normalize_finite_f32(&mut vector, 64.0);

        for &value in vector.iter() {
            assert!(value >= 1.0 / 64.0 - 1e-4);
            assert!(value <= 1.0 / 64.0 + 1e-4);
        }
    }

    #[test]
    fn check_nan() {
        let mut vector = vec! [1.0; 64];
        vector[0] = ::std::f32::NAN;
        normalize_finite_f32(&mut vector, 63.0);

        assert!(vector[0].is_nan());
        for &value in vector.iter().skip(1) {
            assert!(value >= 1.0 / 63.0 - 1e-4);
            assert!(value <= 1.0 / 63.0 + 1e-4);
        }
    }

    #[test]
    fn check_inf() {
        let mut vector = vec! [1.0; 64];
        vector[0] = ::std::f32::INFINITY;
        normalize_finite_f32(&mut vector, 63.0);

        assert_eq!(vector[0], ::std::f32::INFINITY);
        for &value in vector.iter().skip(1) {
            assert!(value >= 1.0 / 63.0 - 1e-4);
            assert!(value <= 1.0 / 63.0 + 1e-4);
        }
    }

    #[test]
    fn check_neg_inf() {
        let mut vector = vec! [1.0; 64];
        vector[0] = ::std::f32::NEG_INFINITY;
        normalize_finite_f32(&mut vector, 63.0);

        assert_eq!(vector[0], ::std::f32::NEG_INFINITY);
        for &value in vector.iter().skip(1) {
            assert!(value >= 1.0 / 63.0 - 1e-4);
            assert!(value <= 1.0 / 63.0 + 1e-4);
        }
    }
}
