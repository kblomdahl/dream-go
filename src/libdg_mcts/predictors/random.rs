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

use rand::{thread_rng, Rng};

use crate::asm::normalize_finite_f32;
use crate::{Predictor, Prediction};
use dg_go::{utils::symmetry, Board, Color};
use dg_utils::types::f16;

/// An implementation of `Predictor` that returns completely random predictions. This
/// is useful for testing purposes.
#[derive(Clone, Default)]
pub struct RandomPredictor;

impl Predictor for RandomPredictor {
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
                Prediction::new(f16::from(value), policy.into_iter().map(|x| f16::from(x)).collect())
            })
            .collect()
    }
}
