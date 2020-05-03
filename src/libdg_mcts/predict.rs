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

use dg_utils::types::f16;

pub trait Predictor : Clone + Send {
    /// Returns the result of the given query.
    ///
    /// # Arguments
    ///
    /// * `features` - the features to query
    ///
    fn predict(&self, features: Vec<f16>) -> Option<(f32, Vec<f32>)>;

    /// Returns the results of the given queries.
    ///
    /// # Arguments
    ///
    /// * `features_list` - the features to query over
    ///
    fn predict_all<E: Iterator<Item=Vec<f16>>>(&self, features_list: E) -> Vec<Option<(f32, Vec<f32>)>>;

    /// waits until all other predicts that are currently running in the
    /// background has finished.
    fn synchronize(&self);
}

/// An implementation of `Predictor` that returns completely random predictions. This
/// is useful for testing purposes.
#[cfg(test)]
#[derive(Clone, Default)]
pub struct RandomPredictor;

#[cfg(test)]
impl Predictor for RandomPredictor {
    fn predict(&self, _features: Vec<f16>) -> Option<(f32, Vec<f32>)> {
        use rand::{thread_rng, Rng};
        use super::asm::normalize_finite_f32;

        let mut policy = vec! [0.0; 368];
        let mut total_policy = 0.0;

        for i in 0..362 {
            let value = thread_rng().gen();

            policy[i] = value;
            total_policy += value;
        }

        normalize_finite_f32(&mut policy, total_policy);
        Some((thread_rng().gen_range(-1.0, 1.0), policy))
    }

    fn predict_all<E: Iterator<Item=Vec<f16>>>(&self, features_list: E) -> Vec<Option<(f32, Vec<f32>)>> {
        features_list.map(|features| self.predict(features)).collect()
    }

    fn synchronize(&self) {
        // pass
    }
}

/// An implementation of `Predict` that always returns the given point as the
/// prediction. This is mainly intended for testing purposes.
#[cfg(test)]
#[derive(Clone, Default)]
pub struct FakePredictor {
    point: usize,
    value: f32
}

#[cfg(test)]
impl FakePredictor {
    pub fn new(point: usize, value: f32) -> Self {
        Self { point, value }
    }
}

#[cfg(test)]
impl Predictor for FakePredictor {
    fn predict(&self, _features: Vec<f16>) -> Option<(f32, Vec<f32>)> {
        let mut policy = vec! [0.0; 368];
        policy[self.point] = 1.0;

        Some((self.value, policy))
    }

    fn predict_all<E: Iterator<Item=Vec<f16>>>(&self, features_list: E) -> Vec<Option<(f32, Vec<f32>)>> {
        features_list.map(|features| self.predict(features)).collect()
    }

    fn synchronize(&self) {
        // pass
    }
}
