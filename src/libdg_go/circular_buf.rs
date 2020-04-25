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

/// 
pub struct CircularIterator<'a, T: Sized + Copy + Default> {
    count: usize,
    position: usize,
    buf: &'a [T]
}

/// Lookup table computing `(index + 1) % 6`.
const N_MOD_SIX: [usize; 6] = [1, 2, 3, 4, 5, 0];

/// Lookup table computing `(index - 1) % 6` with wrap-around for negative
/// numbers.
const P_MOD_SIX: [usize; 6] = [5, 0, 1, 2, 3, 4];

impl<'a, T: Sized + Copy + Default> Iterator for CircularIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 6 {
            None
        } else {
            let index = self.position;
            self.position = P_MOD_SIX[self.position];
            self.count += 1;

            Some(self.buf[index])
        }
    }
}

/// A circular stack that keeps track of the six most recent pushed buffers.
pub struct CircularBuf<T: Sized + Copy + Default> {
    position: usize,
    buf: [T; 6]
}

impl<T: Sized + Copy + Default> Clone for CircularBuf<T> {
    fn clone(&self) -> Self {
        CircularBuf {
            position: self.position,
            buf: self.buf
        }
    }
}

impl<T: Sized + Copy + Default> CircularBuf<T> {
    pub fn new() -> Self {
        CircularBuf {
            position: 0,
            buf: [T::default(); 6]
        }
    }

    /// Adds another value to this circular buffer.
    /// 
    /// # Arguments
    /// 
    /// * `value` - 
    /// 
    pub fn push(&mut self, value: T) {
        self.buf[self.position] = value;
        self.position = N_MOD_SIX[self.position];
    }

    /// Returns an iterator over all the buffers in the stack starting with the
    /// most recent one, and going backward in time.
    pub fn iter(&self) -> CircularIterator<'_, T> {
        CircularIterator {
            count: 0,
            position: P_MOD_SIX[self.position],
            buf: &self.buf
        }
    }
}

#[cfg(test)]
mod tests {
    use circular_buf::*;

    #[test]
    fn check() {
        let mut buf = CircularBuf::new();

        buf.push(0);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4);
        buf.push(5);
        buf.push(6);
        buf.push(7);
        buf.push(8);

        let mut iter = buf.iter();

        assert_eq!(iter.next().unwrap(), 8);
        assert_eq!(iter.next().unwrap(), 7);
        assert_eq!(iter.next().unwrap(), 6);
        assert_eq!(iter.next().unwrap(), 5);
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.next().unwrap(), 3);
        assert!(iter.next().is_none());
    }
}
