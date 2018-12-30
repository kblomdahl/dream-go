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

/// Returns true if this set contains the given value (using SSE2, and AVX2
/// instructions).
///
/// # Arguments
///
/// * `other` - the value to look for
///
#[target_feature(enable = "sse2,avx,avx2")]
unsafe fn _contains_u64x16_avx2(haystack: &[u64], needle: u64) -> bool {
    let mut haystack = haystack.as_ptr();

    // unroll the entire loop for a better pipeline
    let needle = _mm_set1_epi64x(needle as i64);

    for _i in 0..2 {
        let a = _mm_loadu_si128(haystack.add(0) as *const _);
        let b = _mm_loadu_si128(haystack.add(2) as *const _);
        let c = _mm_loadu_si128(haystack.add(4) as *const _);
        let d = _mm_loadu_si128(haystack.add(6) as *const _);
        let eq_a = _mm_cmpeq_epi64(a, needle);
        let eq_b = _mm_cmpeq_epi64(b, needle);
        let eq_c = _mm_cmpeq_epi64(c, needle);
        let eq_d = _mm_cmpeq_epi64(d, needle);

        // we do not care where the needle was found, just if it is present so just
        // squash all of the elements together.
        let or_ab = _mm_or_si128(eq_a, eq_b);
        let or_cd = _mm_or_si128(eq_c, eq_d);
        let or = _mm_or_si128(or_ab, or_cd);

        if _mm_movemask_epi8(or) != 0 {
            return true
        }

        haystack = haystack.add(8);
    }

    false
}

/// Returns true if this set contains the given value.
///
/// # Arguments
///
/// * `other` - the value to look for
///
#[inline(always)]
pub fn contains_u64x16(haystack: &[u64], needle: u64) -> bool {
    debug_assert_eq!(haystack.len(), 16);

    if is_x86_feature_detected!("avx2") {
        unsafe { _contains_u64x16_avx2(haystack, needle) }
    } else {
        (0..16).any(|x| haystack[x] == needle)
    }
}
