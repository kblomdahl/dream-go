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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use parallel::{self, OneSender};
use go::util::features::{FEATURE_SIZE};
use nn::devices::{DEVICES, set_current_device};
use nn::{self, Network, Output, OutputSet, Workspace};
use util::config;

pub type PredictGuard<'a> = parallel::ServiceGuard<'a, PredictState>;
pub type PredictService = parallel::Service<PredictState>;

pub fn service(network: Network) -> PredictService {
    PredictService::new(None, PredictState::new(network))
}

pub enum PredictRequest {
    /// Request to compute the value and policy for some feature.
    Ask(Vec<i8>),

    /// Indicate that a worker is waiting for some other thread to finish
    /// and should be awaken after the next batch of computations finish.
    Wait
}

pub struct PredictState {
    /// The neural network weights
    network: Network,

    /// The number of requests that are being processed by the GPU at
    /// this moment
    running_count: AtomicUsize,

    /// The features to get the value and policy for.
    features_list: Vec<i8>,

    /// The sender to response to each of the features in `features_list`
    /// over.
    sender_list: Vec<OneSender<Option<(f32, Vec<f32>)>>>,

    /// All threads that want to get notified when something changed.
    waiting_list: Vec<OneSender<Option<(f32, Vec<f32>)>>>,
}

impl PredictState {
    pub fn new(network: Network) -> PredictState {
        PredictState {
            network: network,
            running_count: AtomicUsize::new(0),
            features_list: vec! [],
            sender_list: vec! [],
            waiting_list: vec! []
        }
    }

    /// Returns the network used to perform the predictions.
    pub fn get_network(&self) -> &Network {
        &self.network
    }

    /// Run the `nn::forward` function for the given features and wrap the
    /// results into `Array` elements. This version assumes the neural network
    /// use `f32` weights.
    /// 
    /// # Arguments
    /// 
    /// * `workspace` - 
    /// * `features_list` - 
    /// 
    fn forward(workspace: &mut Workspace, features_list: &[i8]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut outputs = nn::forward(
            workspace,
            features_list,
            OutputSet::default().with(Output::Policy).with(Output::Value)
        );

        let value_list = outputs.take(Output::Value);
        let policy_list = outputs.take(Output::Policy).chunks(362)
            .map(|p| p.to_vec())
            .collect();

        (value_list, policy_list)
    }

    fn predict(
        state: &Mutex<PredictState>,
        mut state_lock: MutexGuard<PredictState>,
        batch_size: usize
    )
    {
        let num_items = state_lock.sender_list.len();
        let split_index = num_items - batch_size;
        let features_list = state_lock.features_list.split_off(split_index * FEATURE_SIZE);
        let sender_list = state_lock.sender_list.split_off(split_index);
        let network = state_lock.network.clone();  // just a bunch of Arc<...> so cheap to clone

        // keep track of the number of running evaluations so that we avoid
        // running duplicate small evaluations instead of one large one
        state_lock.running_count.fetch_add(1, Ordering::SeqCst);
        drop(state_lock);

        debug_assert!(features_list.len() == batch_size * FEATURE_SIZE);
        debug_assert!(sender_list.len() == batch_size);

        // perform the neural network predictions and then inform all of
        // the receivers
        let (value_list, policy_list) = PredictState::forward(
            &mut network.get_workspace(batch_size),
            &features_list
        );

        // send out our predictions to all of the receivers
        let response_iter = value_list.into_iter().zip(policy_list.into_iter());

        for (sender, response) in sender_list.into_iter().zip(response_iter) {
            sender.send(Some(response));
        }

        // wake up all of the receivers that are waiting for something to change
        let mut state_lock = state.lock().unwrap();
        let num_waiting = state_lock.waiting_list.len();

        for waiting in state_lock.waiting_list.drain(0..num_waiting) {
            waiting.send(None);
        }

        // decrease the number of running neural network evaluations
        state_lock.running_count.fetch_sub(1, Ordering::SeqCst);
    }

    fn check(
        state: &Mutex<PredictState>,
        mut state_lock: MutexGuard<PredictState>,
        has_more: bool
    )
    {
        let num_requests = state_lock.sender_list.len();
        let batch_size = *config::BATCH_SIZE;

        if has_more {
            if num_requests >= batch_size {
                // the batch is full, start an evaluation
                PredictState::predict(state, state_lock, batch_size);
            } else {
                // more requests are incoming, wait for them before trying to
                // evaluate a batch
            }
        } else if num_requests > 0 {
            assert!(num_requests <= batch_size);

            // immediately evaluate when we hit a barrier in order to:
            //   1. minimize the latency between request and response
            //   2. avoid a race condition where a request that arrives
            //      during an evaluation does not trigger one.
            PredictState::predict(state, state_lock, num_requests);
        } else if state_lock.running_count.load(Ordering::SeqCst) == 0 {
            // everything is asleep? probably a race condition between the
            // pending message being sent and it being received. Just wake
            // everything up and it should normalize.
            let num_waiting = state_lock.waiting_list.len();

            for waiting in state_lock.waiting_list.drain(0..num_waiting) {
                waiting.send(None);
            }
        } else {
            // wait until the currently running request finish instead of
            // waking up any threads
        }
    }
}

impl parallel::ServiceImpl for PredictState {
    type State = PredictState;
    type Request = PredictRequest;
    type Response = Option<(f32, Vec<f32>)>;

    fn get_thread_count() -> usize {
        let num_devices = DEVICES.len();
        let num_busy = *config::NUM_THREADS / *config::BATCH_SIZE;

        ::std::cmp::max(::std::cmp::max(2, num_devices), num_busy)
    }

    fn setup_thread(index: usize) {
        let device_id = DEVICES[index % DEVICES.len()];

        set_current_device(device_id);
    }

    fn check_sleep(state: MutexGuard<Self::State>) {
        let num_requests = state.sender_list.len();
        let num_waiting = state.waiting_list.len();

        assert_eq!(num_requests, 0, "we should never sleep with a pending request -- {}", num_requests);
        assert_eq!(num_waiting, 0, "we should never sleep with a pending wait -- {}", num_waiting);
    }

    fn process(
        state: &Mutex<Self::State>,
        mut state_lock: MutexGuard<Self::State>,
        req: Self::Request,
        sender: OneSender<Self::Response>,
        has_more: bool
    )
    {
        match req {
            PredictRequest::Ask(features) => {
                state_lock.features_list.extend_from_slice(&features);
                state_lock.sender_list.push(sender);
            },
            PredictRequest::Wait => {
                state_lock.waiting_list.push(sender);
            }
        };

        PredictState::check(state, state_lock, has_more);
    }
}
