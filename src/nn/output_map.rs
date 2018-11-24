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
    ValueGemm = 5,

    Residual_00 = 6,
    Residual_01 = 7,
    Residual_02 = 8,
    Residual_03 = 9,
    Residual_04 = 10,
    Residual_05 = 11,
    Residual_06 = 12,
    Residual_07 = 13,
    Residual_08 = 14,
    Residual_09 = 15,
    Residual_10 = 16,
    Residual_11 = 17,
    Residual_12 = 18,
    Residual_13 = 19,
    Residual_14 = 20,
    Residual_15 = 21,
    Residual_16 = 22,
    Residual_17 = 23,
    Residual_18 = 24,
    Residual_19 = 25,
    Residual_20 = 26,
    Residual_21 = 27,
    Residual_22 = 28,
    Residual_23 = 29,
    Residual_24 = 30,
    Residual_25 = 31,
    Residual_26 = 32,
    Residual_27 = 33,
    Residual_28 = 34,
    Residual_29 = 35,
    Residual_30 = 36,
    Residual_31 = 37,
    Residual_32 = 38,
    Residual_33 = 39,
    Residual_34 = 40,
    Residual_35 = 41,
    Residual_36 = 42,
    Residual_37 = 43,
    Residual_38 = 44,
    Residual_39 = 45,
}

/// The total number of elements in the `Output` enum.
const OUTPUT_SIZE: usize = 46;

pub struct OutputMap<T> {
    array: [Option<T>; OUTPUT_SIZE]
}

impl<T> OutputMap<T> {
    pub fn new() -> OutputMap<T> {
        OutputMap {
            array: [
                None, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None, None,
                None,
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

    pub fn iter(&self) -> OutputSetIter {
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
