// Copyright 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use dg_go::utils::symmetry::Transform;
use dg_go::{Board, Color, Point};
use dg_utils::types::f16;

#[derive(Clone)]
pub struct Prediction {
    value: f16,
    policy: Vec<f16>
}

impl Prediction {
    pub fn new(value: f16, policy: Vec<f16>) -> Self {
        Self { value, policy }
    }

    pub fn with_transform(other: &Self, transform: Transform) -> Self {
        let t_table = transform.get_table();
        let mut remapped_policy = vec! [f16::from(0.0); 362];

        for i in Point::all() {
            let j = t_table[i].to_packed_index();
            remapped_policy[j] = other.policy[i.to_packed_index()];
        }
        remapped_policy[Point::default().to_packed_index()] = other.policy[Point::default().to_packed_index()];

        Self {
            value: other.value,
            policy: remapped_policy
        }
    }

    pub fn value(&self) -> f32 {
        f32::from(self.value)
    }

    pub fn winrate(&self) -> f32 {
        0.5 * self.value() + 0.5
    }

    pub fn policy(&self) -> Vec<f32> {
        self.policy.iter().map(|&x| f32::from(x)).collect()
    }
}

pub trait Predictor : Send {
    /// Returns the maximum number of parallel calls that should be made into
    /// `predict`.
    fn max_num_threads(&self) -> usize;

    /// Retrieve the value and policy from the transposition table.
    ///
    /// # Arguments
    ///
    /// * `board` - the board to get from the table
    /// * `to_move` - the color to get from the table
    /// * `symmetry` - the symmetry to get from the table
    ///
    fn fetch(&self, board: &Board, to_move: Color, symmetry: Transform) -> Option<Prediction>;

    /// Adds the given value and policy to the transposition table.
    ///
    /// # Arguments
    ///
    /// * `board` - the board to add to the table
    /// * `to_move` - the color to add to the table
    /// * `symmetry` - the symmetry to add to the table
    /// * `response` - the response to add to the table
    ///
    fn cache(&self, board: &Board, to_move: Color, symmetry: Transform, response: Prediction);

    /// Returns the result of the given query.
    ///
    /// # Arguments
    ///
    /// * `features` - the features to query
    ///
    fn predict(&self, features: &[f16], batch_size: usize) -> Vec<Prediction>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_with_transform() {
        let original = Prediction::new(
            f16::from(0.0),
            (0..362).map(|i| f16::from(i as f32)).collect::<Vec<_>>()
        );

        assert_eq!(original.policy()[0], Prediction::with_transform(&original, Transform::Identity).policy()[0]);
        assert_eq!(original.policy()[0], Prediction::with_transform(&original, Transform::Rot180).policy()[360]);
        assert_eq!(original.policy()[361], Prediction::with_transform(&original, Transform::Rot180).policy()[361]);
    }
}