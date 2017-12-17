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

/// A LRA set that only keeps the eight most recently added values.
#[derive(Clone)]
pub struct SmallSet {
    buf: [u64; 8],
    count: usize
}

impl SmallSet {
    /// Returns an empty set.
    pub fn new() -> SmallSet {
        SmallSet { buf: [0; 8], count: 0 }
    }

    /// Adds the given value to this set, removing the oldest value if the set overflows.
    /// 
    /// # Arguments
    /// 
    /// * `value` - the value to add to the set
    /// 
    pub fn push(&mut self, value: u64) {
        self.buf[self.count] = value;
        self.count += 1;

        if self.count == 8 {
            self.count = 0;
        }
    }

    /// Returns true if this set contains the given value.
    /// 
    /// # Arguments
    /// 
    /// * `other` - the value to look for
    /// 
    pub fn contains(&self, other: u64) -> bool {
        (0..8).any(|x| self.buf[x] == other)
    }

    /// Returns an iterator over all elements in this set.
    pub fn iter<'a>(&'a self) -> SmallIter<'a> {
        SmallIter {
            set: self,
            position: 0
        }
    }
}

/// Iterator over all elements contained within a `SmallSet`.
pub struct SmallIter<'a> {
    set: &'a SmallSet,
    position: usize
}

impl<'a> Iterator for SmallIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.position >= 8 {
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
    use go::small_set::*;

    #[test]
    fn check() {
        let mut s = SmallSet::new();

        s.push(1);
        s.push(2);
        s.push(3);

        assert!(s.contains(1));
        assert!(s.contains(2));
        assert!(s.contains(3));
        assert!(!s.contains(4));
    }
}
