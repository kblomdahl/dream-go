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

use crate::{Predictor, Prediction};
use dg_go::{Board, Color};
use dg_utils::types::f16;

#[derive(Clone, Default)]
pub struct NanPredictor;

impl Predictor for NanPredictor {
    fn max_num_threads(&self) -> usize {
        1
    }

    fn fetch(&self, _board: &Board, _to_move: Color) -> Option<Prediction> {
        None
    }

    fn cache(&self, _board: &Board, _to_move: Color, _response: Prediction) {
        // pass
    }

    fn initial_predict(&self, _features: &[f16], batch_size: usize) -> Vec<Prediction> {
        (0..batch_size)
            .map(|_| {
                Prediction::new(
                    f16::from(0.0),
                    vec! [f16::from(::std::f32::NEG_INFINITY); 362],
                    vec! [f16::from(0.0); 722]
                )
            })
            .collect()
    }

    fn predict(&self, _hidden_states: &[f16], features: &[f16], batch_size: usize) -> Vec<Prediction> {
        self.initial_predict(features, batch_size)
    }
}
