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

use dg_go::utils::symmetry;
use dg_go::{Board, Color};
use dg_utils::types::f16;

#[derive(Clone)]
pub struct PredictResponse {
    value: f16,
    policy: Vec<f16>
}

impl PredictResponse {
    pub fn new(value: f16, policy: Vec<f16>) -> Self {
        Self { value, policy }
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
    fn fetch(&self, board: &Board, to_move: Color, symmetry: symmetry::Transform) -> Option<PredictResponse>;

    /// Adds the given value and policy to the transposition table.
    ///
    /// # Arguments
    ///
    /// * `board` - the board to add to the table
    /// * `to_move` - the color to add to the table
    /// * `symmetry` - the symmetry to add to the table
    /// * `response` - the response to add to the table
    ///
    fn cache(&self, board: &Board, to_move: Color, symmetry: symmetry::Transform, response: PredictResponse);

    /// Returns the result of the given query.
    ///
    /// # Arguments
    ///
    /// * `features` - the features to query
    ///
    fn predict(&self, features: &[f16], batch_size: usize) -> Vec<PredictResponse>;
}

/// An implementation of `Predictor` that returns completely random predictions. This
/// is useful for testing purposes.
#[derive(Clone, Default)]
pub struct RandomPredictor;

impl Predictor for RandomPredictor {
    fn max_num_threads(&self) -> usize {
        1
    }

    fn fetch(&self, _board: &Board, _to_move: Color, _symmetry: symmetry::Transform) -> Option<PredictResponse> {
        None
    }

    fn cache(&self, _board: &Board, _to_move: Color, _symmetry: symmetry::Transform, _response: PredictResponse) {
        // pass
    }

    fn predict(&self, _features: &[f16], batch_size: usize) -> Vec<PredictResponse> {
        use rand::{thread_rng, Rng};
        use super::asm::normalize_finite_f32;

        (0..batch_size)
            .map(|_| {
                let value = thread_rng().gen_range(-1.0..1.0);
                let mut policy = vec! [0.0; 368];
                let mut total_policy = 0.0;

                for i in 0..362 {
                    let value = thread_rng().gen();

                    policy[i] = value;
                    total_policy += value;
                }

                normalize_finite_f32(&mut policy, total_policy);
                PredictResponse::new(f16::from(value), policy.into_iter().map(|x| f16::from(x)).collect())
            })
            .collect()
    }
}

/// An implementation of `Predict` that always returns the given point as the
/// prediction. This is mainly intended for testing purposes.
#[cfg(test)]
#[derive(Clone, Default)]
pub struct FakePredictor {
    point: usize,
    value: f16
}

#[cfg(test)]
impl FakePredictor {
    pub fn new(point: usize, value: f32) -> Self {
        Self { point: point, value: f16::from(value) }
    }
}

#[cfg(test)]
impl Predictor for FakePredictor {
    fn max_num_threads(&self) -> usize {
        1
    }

    fn fetch(&self, _board: &Board, _to_move: Color, _symmetry: symmetry::Transform) -> Option<PredictResponse> {
        None
    }

    fn cache(&self, _board: &Board, _to_move: Color, _symmetry: symmetry::Transform, _response: PredictResponse) {
        // pass
    }

    fn predict(&self, _features: &[f16], batch_size: usize) -> Vec<PredictResponse> {
        let mut policy = vec! [f16::from(0.0); 368];
        policy[self.point] = f16::from(1.0);

        (0..batch_size)
            .map(|_| PredictResponse::new(self.value, policy.clone()))
            .collect()
    }
}
