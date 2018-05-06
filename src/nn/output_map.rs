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

#[repr(u8)]
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
pub enum Output {
    Policy = 0,  // the final policy output
    Value = 1,   // the final value output

    Upsample = 2,
    PolicyDown = 3,
    ValueDown = 4,

    Residual_00 = 5,
    Residual_01 = 6,
    Residual_02 = 7,
    Residual_03 = 8,
    Residual_04 = 9,
    Residual_05 = 10,
    Residual_06 = 11,
    Residual_07 = 12,
    Residual_08 = 13,
}

/// The total number of elements in the `Output` enum.
const OUTPUT_SIZE: usize = 14;

pub struct OutputMap<T> {
    array: [Option<T>; OUTPUT_SIZE]
}

impl<T> OutputMap<T> {
    pub fn new() -> OutputMap<T> {
        OutputMap {
            array: [
                None, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None,
            ]
        }
    }

    pub fn put(&mut self, key: Output, value: T) {
        self.array[key as usize] = Some(value);
    }

    pub fn with(mut self, key: Output, value: T) -> OutputMap<T> {
        self.put(key, value);
        self
    }

    pub fn get(&mut self, key: Output) -> &T {
        self.array[key as usize].as_ref().unwrap()
    }

    pub fn take(&mut self, key: Output) -> T {
        self.array[key as usize].take().unwrap()
    }
}

pub struct OutputSet {
    array: [bool; OUTPUT_SIZE]
}

impl OutputSet {
    pub fn new() -> OutputSet {
        OutputSet {
            array: [false; OUTPUT_SIZE]
        }
    }

    pub fn add(&mut self, key: Output) {
        self.array[key as usize] = true;
    }

    pub fn with(mut self, key: Output) -> OutputSet {
        self.add(key);
        self
    }

    pub fn contains(&self, key: Output) -> Option<Output> {
        if self.array[key as usize] {
            Some(key)
        } else {
            None
        }
    }

    pub fn iter<'a>(&'a self) -> OutputSetIter<'a> {
        OutputSetIter {
            array: &self.array,
            position: 0
        }
    }
}

pub struct OutputSetIter<'a> {
    array: &'a [bool; OUTPUT_SIZE],
    position: usize
}

impl<'a> Iterator for OutputSetIter<'a> {
    type Item = Output;

    fn next(&mut self) -> Option<Output> {
        let i = self.position;

        if i >= self.array.len() {
            None
        } else if self.array[i] {
            self.position += 1;

            Some(unsafe { ::std::mem::transmute::<_, Output>(i as u8) })
        } else {
            self.position += 1;
            self.next()
        }
    }
}
