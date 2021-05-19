// Copyright 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use crate::{Predictor, Prediction};
use dg_go::{utils::symmetry, Board, Color};
use dg_utils::types::f16;

/// An implementation of `Predict` that always returns the given point as the
/// prediction. This is mainly intended for testing purposes.
#[derive(Clone, Default)]
pub struct FakePredictor {
    point: usize,
    value: f16
}

impl FakePredictor {
    pub fn new(point: usize, value: f32) -> Self {
        Self { point: point, value: f16::from(value) }
    }
}

impl Predictor for FakePredictor {
    fn max_num_threads(&self) -> usize {
        1
    }

    fn fetch(&self, _board: &Board, _to_move: Color, _symmetry: symmetry::Transform) -> Option<Prediction> {
        None
    }

    fn cache(&self, _board: &Board, _to_move: Color, _symmetry: symmetry::Transform, _response: Prediction) {
        // pass
    }

    fn predict(&self, _features: &[f16], batch_size: usize) -> Vec<Prediction> {
        let mut policy = vec! [f16::from(0.0); 368];
        policy[self.point] = f16::from(1.0);

        (0..batch_size)
            .map(|_| Prediction::new(self.value, policy.clone()))
            .collect()
    }
}
