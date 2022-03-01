// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use dg_utils::types::f16;
use dg_go::utils::features::{self as features};

use libc::{c_int};

#[repr(C)]
pub struct Example {
    /// The input to the _representation_ model, denoted as `x_t` in the
    /// research paper.
    pub features: *mut f16,
    pub features_shape: [c_int; 5],

    /// The input to the _recurrent_ / _dynamics_ model together with the
    /// previous hidden state, denoted as `a_t` in the research paper.
    pub motion_features: *mut f16,
    pub motion_features_shape: [c_int; 5],

    /// Leela-zero style features for semi-supervised learning.
    pub lz_features: *mut f16,
    pub lz_features_shape: [c_int; 5],

    /// Additional targets to predict from the hidden state during
    /// self-supervised learning, denoted as `x̂_t` in the research paper.
    pub additional_targets: *mut f32,
    pub additional_targets_mask: *mut f32,
    pub additional_targets_shape: [c_int; 5],

    /// The true value of each example, denoted as `v_t` in the research
    /// paper.
    pub value: *mut f32,
    pub value_shape: [c_int; 3],

    /// The true policy of each example. denoted as `a_t` in the research
    /// paper.
    pub policy: *mut f32,
    pub policy_shape: [c_int; 3]
}

impl Example {
    pub fn has_shape(&self) -> bool {
        self.features_shape.iter().product::<i32>() > 0 &&
            self.motion_features_shape.iter().product::<i32>() > 0 &&
            self.additional_targets_shape.iter().product::<i32>() > 0 &&
            self.lz_features_shape.iter().product::<i32>() > 0 &&
            self.value_shape.iter().product::<i32>() > 0 &&
            self.policy_shape.iter().product::<i32>() > 0
    }

    pub fn has_ptr(&self) -> bool {
        !self.features.is_null() &&
        !self.motion_features.is_null() &&
        !self.additional_targets.is_null() &&
        !self.additional_targets_mask.is_null() &&
        !self.lz_features.is_null() &&
        !self.value.is_null() &&
        !self.policy.is_null()
    }

    pub fn features(&mut self, i: usize, j: usize) -> &mut [f16] {
        unsafe {
            ::std::slice::from_raw_parts_mut(
                self.features.offset(tensor_offset(&self.features_shape, &[i as c_int, j as c_int, 0, 0, 0])),
                (self.features_shape[2] * self.features_shape[3] * self.features_shape[4]) as usize
            )
        }
    }

    pub fn motion_features(&mut self, i: usize, j: usize) -> &mut [f16] {
        unsafe {
            ::std::slice::from_raw_parts_mut(
                self.motion_features.offset(tensor_offset(&self.motion_features_shape, &[i as c_int, j as c_int, 0, 0, 0])),
                (self.motion_features_shape[2] * self.motion_features_shape[3] * self.motion_features_shape[4]) as usize
            )
        }
    }

    pub fn additional_targets(&mut self, i: usize, j: usize) -> &mut [f32] {
        unsafe {
            ::std::slice::from_raw_parts_mut(
                self.additional_targets.offset(tensor_offset(&self.additional_targets_shape, &[i as c_int, j as c_int, 0, 0, 0])),
                (self.additional_targets_shape[2] * self.additional_targets_shape[3] * self.additional_targets_shape[4]) as usize
            )
        }
    }

    pub fn additional_targets_mask(&mut self, i: usize, j: usize) -> &mut [f32] {
        unsafe {
            ::std::slice::from_raw_parts_mut(
                self.additional_targets_mask.offset(
                    tensor_offset(
                        &[
                            self.additional_targets_shape[0],
                            self.additional_targets_shape[1],
                            self.additional_targets_shape[4],
                        ],
                        &[i as c_int, j as c_int, 0]
                    )
                ),
                self.additional_targets_shape[4] as usize
            )
        }
    }

    pub fn lz_features(&mut self, i: usize, j: usize) -> &mut [f16] {
        unsafe {
            ::std::slice::from_raw_parts_mut(
                self.lz_features.offset(tensor_offset(&self.lz_features_shape, &[i as c_int, j as c_int, 0, 0, 0])),
                (self.lz_features_shape[2] * self.lz_features_shape[3] * self.lz_features_shape[4]) as usize
            )
        }
    }

    pub fn value(&mut self, i: usize, j: usize) -> &mut [f32] {
        unsafe {
            ::std::slice::from_raw_parts_mut(
                self.value.offset(tensor_offset(&self.value_shape, &[i as c_int, j as c_int, 0])),
                self.value_shape[2] as usize
            )
        }
    }

    pub fn policy(&mut self, i: usize, j: usize) -> &mut [f32] {
        unsafe {
            ::std::slice::from_raw_parts_mut(
                self.policy.offset(tensor_offset(&self.policy_shape, &[i as c_int, j as c_int, 0])),
                self.policy_shape[2] as usize
            )
        }
    }
}

fn tensor_offset(shape: &[c_int], index: &[c_int]) -> isize {
    debug_assert_eq!(shape.len(), index.len());

    let mut offset = 0;
    let mut stride = 1;

    for i in (0..shape.len()).rev() {
        offset += stride * index[i];
        stride *= shape[i];
    }

    offset as isize
}

#[no_mangle]
extern "C" fn set_example_shape(example: *mut Example, num_examples: c_int, num_unrolls: c_int) -> c_int {
    if example.is_null() {
        return -14; // EFAULT
    }

    let num_features = features::Default::num_features();
    let num_additional_features = features::Default::num_additional_features();
    let num_motion_features = features::Default::num_motion_features();
    let example = unsafe { &mut (*example) };

    example.features_shape = [num_examples, num_unrolls, 19, 19, num_features as c_int];
    example.motion_features_shape = [num_examples, num_unrolls, 19, 19, num_motion_features as c_int];
    example.additional_targets_shape = [num_examples, num_unrolls, 19, 19, num_additional_features as c_int];
    example.lz_features_shape = [num_examples, num_unrolls, 19, 19, 18];
    example.value_shape = [num_examples, num_unrolls, 1];
    example.policy_shape = [num_examples, num_unrolls, 362];

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_index_stride() {
        assert_eq!(tensor_offset(&[2, 8, 362], &[0, 0, 0]), 0);
        assert_eq!(tensor_offset(&[2, 8, 362], &[0, 1, 0]), 362);
        assert_eq!(tensor_offset(&[2, 8, 362], &[1, 7, 0]), 5430);
    }
}
