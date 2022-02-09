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

use crate::{Predictor, Prediction, NodeTrace};
use super::shared_context::SearchContext;
use dg_go::utils::features::{self, HWC, Features};
use dg_go::utils::symmetry;
use dg_go::Board;
use dg_utils::types::f16;

use std::sync::Arc;

#[derive(Clone)]
pub enum EventKind {
    Predict(Vec<f16>, Vec<f16>),
    Insert(Prediction),
    Pending
}

#[derive(Clone)]
pub struct Event {
    pub kind: EventKind,
    pub search_context: Arc<SearchContext>,
    pub board: Board,
    pub trace: NodeTrace
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    pub fn predict(server: &Box<dyn Predictor + Sync>, search_context: Arc<SearchContext>, board: Board, trace: NodeTrace) -> Self {
        let &(_, last_move, _, hidden_states) = trace.last().unwrap();
        let to_move = last_move.opposite();
        let kind =
            if let Some(response) = server.fetch(&board, to_move) {
                EventKind::Insert(response)
            } else {
                let features = features::Default::new(&board).get_features::<HWC, f16>(to_move, symmetry::Transform::Identity);
                EventKind::Predict(unsafe { (*hidden_states).clone() }, features)
            };

        Self { kind, search_context, board, trace }
    }

    pub fn into_insert(mut self, response: Prediction) -> (EventKind, Event) {
        let prev_kind = self.kind;
        self.kind = EventKind::Insert(response);
        (prev_kind, self)
    }

    pub fn into_pending(mut self) -> (EventKind, Event) {
        let prev_kind = self.kind;
        self.kind = EventKind::Pending;
        (prev_kind, self)
    }
}
