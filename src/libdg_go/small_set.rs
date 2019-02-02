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

use asm::contains_u64x16;

const SET_SIZE: usize = 16;

/// A LRA set that only keeps the eight most recently added values.
#[derive(Clone)]
#[repr(align(16))]
pub struct SmallSet64 {
    buf: [u64; SET_SIZE],
    count: usize
}

impl SmallSet64 {
    /// Returns an empty set.
    pub fn new() -> SmallSet64 {
        SmallSet64 { buf: [0; SET_SIZE], count: 0 }
    }

    /// Adds the given value to this set, removing the oldest value if
    /// the set overflows.
    ///
    /// # Arguments
    ///
    /// * `value` - the value to add to the set
    ///
    pub fn push(&mut self, value: u64) {
        self.buf[self.count] = value;
        self.count += 1;

        if self.count == SET_SIZE {
            self.count = 0;
        }
    }

    /// Returns true if this set contains the given value.
    ///
    /// # Arguments
    ///
    /// * `other` - the value to look for
    ///
    #[inline(always)]
    pub fn contains(&self, other: u64) -> bool {
        contains_u64x16(&self.buf, other)
    }

    /// Returns an iterator over all elements in this set.
    pub fn iter(&self) -> SmallIter64 {
        SmallIter64 {
            set: self,
            position: 0
        }
    }
}

/// Iterator over all elements contained within a `SmallSet64`.
pub struct SmallIter64<'a> {
    set: &'a SmallSet64,
    position: usize
}

impl<'a> Iterator for SmallIter64<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.position >= SET_SIZE {
            None
        } else {
            let value = self.set.buf[self.position];
            self.position += 1;

            Some(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use small_set::*;
    use test::Bencher;

    #[test]
    fn check_64() {
        let mut s = SmallSet64::new();

        s.push(1);
        s.push(2);
        s.push(3);

        assert!(s.contains(1));
        assert!(s.contains(2));
        assert!(s.contains(3));
        assert!(!s.contains(4));
    }

    #[bench]
    fn contains_64(b: &mut Bencher) {
        let mut s = SmallSet64::new();

        s.push(1);
        s.push(2);
        s.push(3);

        b.iter(|| {
            assert_eq!(s.contains(8), false);
        });
    }
}
