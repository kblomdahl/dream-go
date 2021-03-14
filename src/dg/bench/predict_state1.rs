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

use dg_mcts::predict_service::{PredictState, PredictRequest};
use dg_mcts::parallel;

use crossbeam_channel::Sender;
use std::sync::{Mutex, MutexGuard};

pub struct PredictState1;

impl parallel::ServiceImpl for PredictState1 {
    type State = PredictState;
    type Request = PredictRequest;
    type Response = Option<(f32, Vec<f32>)>;

    fn get_thread_count() -> usize {
        1
    }

    fn setup_thread(index: usize) {
        PredictState::setup_thread(index)
    }

    fn process(
        state: &Mutex<Self::State>,
        state_lock: MutexGuard<Self::State>,
        req: Self::Request,
        sender: Sender<Self::Response>,
        has_more: bool
    )
    {
        PredictState::process(state, state_lock, req, sender, has_more)
    }
}
