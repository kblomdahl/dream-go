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
#[inline]
pub fn count_zeros(array: &[u8]) -> usize {
    debug_assert!(array.len() == 384);

    let count: usize;

    if cfg!(target_arch = "x86_64") {
        unsafe {
            // 256-bit AVX2 implementation of the following algorithm, where
            // the main _trick_ is the use of the `movemask` and `popcnt` in
            // order to figure out the number of values in each lane that
            // succeeded the test.
            //
            // ```
            // let mut count = 0
            // for i in 0..361 {
            //     if array[i] == 1 { count += 1 }
            // }
            // count
            // ```
            asm!(
                r#"
                mov rcx, 12                # set loop counter
                vpxor ymm0, ymm0, ymm0     # set ymm0 = 0
                xor rax, rax               # set rax = 0
                xor rdx, rdx               # set rdx = 0

                1:
                vmovups ymm1, [rbx]        # ymm1 = array[rbx..(rbx+32)]
                vpcmpeqb ymm1, ymm1, ymm0  # ymm1 = (xmm1 == 0)
                vpmovmskb edx, ymm1        # edx = 1 bit set for each byte in ymm1 that is not 0
                popcnt edx, edx            # edx = number of bits set in edx
                add rax, rdx               # rax += edx

                add rbx, 32                # rbx += 32
                dec ecx                    # ecx -= 1
                jnz 1b                     # repeat if ecx > 0
                "#
                : "={rax}"(count)  // outputs
                : "{rbx}"(&array[0])  // inputs
                : "rax", "rbx", "rcx", "rdx", "ymm0", "ymm1"  // clobbers
                : "intel", "volatile"
            );
        }
    } else {
        count = (0..361).filter(|&i| array[i] == 0).count();
    }

    count
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
