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
use nn::{self, Network, Type, TYPE, Workspace};
use util::array::*;
use util::config;
use util::singleton::*;
use util::types::*;

pub type PredictGuard<'a> = parallel::ServiceGuard<'a, PredictState>;
pub type PredictService = parallel::Service<PredictState>;

pub fn service(network: Network) -> PredictService {
    PredictService::new(None, PredictState::new(network))
}

pub enum PredictRequest {
    /// Request to compute the value and policy for some feature.
    Ask(Array),

    /// Indicate that a worker is waiting for some other thread to finish
    /// and should be awaken after the next batch of computations finish.
    Wait
}

struct PredictShared {
    /// The features to get the value and policy for.
    features_list: Vec<Array>,

    /// The sender to response to each of the features in `features_list`
    /// over.
    sender_list: Vec<OneSender<Option<(Singleton, Array)>>>,

    /// All threads that want to get notified when something changed.
    waiting_list: Vec<OneSender<Option<(Singleton, Array)>>>
}

pub struct PredictState {
    network: Network,
    shared: Mutex<PredictShared>,

    /// The number of requests that are being processed by the GPU at
    /// this moment
    running_count: AtomicUsize,
}

impl PredictState {
    pub fn new(network: Network) -> PredictState {
        PredictState {
            network: network,
            shared: Mutex::new(PredictShared {
                features_list: vec! [],
                sender_list: vec! [],
                waiting_list: vec! []
            }),

            running_count: AtomicUsize::new(0)
        }
    }

    /// Returns the network used to perform the predictions.
    pub fn get_network<'a>(&'a self) -> &'a Network {
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
    fn forward<T, R>(
        workspace: &mut Workspace,
        features_list: Vec<Array>
    ) -> (Vec<Singleton>, Vec<Array>)
        where T: From<f32> + Clone,
              R: From<f32> + Clone, Box<[R]>: Into<Array>,
              Array: Into<Box<[T]>> + From<Box<[R]>>,
              Singleton: From<R>,
    {
        let (value_list, policy_list) = nn::forward::<T, R>(
            workspace,
            &features_list.into_iter()
                .map(|feature| Array::into(feature))
                .collect()
        );

        // wrap the results in `Array` so that we can avoid having to pass
        // generics everywhere
        let value_list = value_list.into_iter()
            .map(|value| Singleton::from(value))
            .collect();
        let policy_list = policy_list.into_iter()
            .map(|policy| Array::from(policy))
            .collect();

        (value_list, policy_list)
    }

    fn predict(&self, mut shared: MutexGuard<PredictShared>, batch_size: usize) {
        let num_items = shared.features_list.len();
        let features_list = shared.features_list.split_off(num_items - batch_size);
        let sender_list = shared.sender_list.split_off(num_items - batch_size);

        // get ride of our MutexGuard to `shared` to allow for parallel execution
        // while we are busy running the forward pass through the network.
        drop(shared);

        debug_assert!(features_list.len() == batch_size);
        debug_assert!(sender_list.len() == batch_size);

        // perform the neural network predictions and then inform all of
        // the receivers
        let mut workspace = self.network.get_workspace(batch_size);
        let (value_list, policy_list) = match *TYPE {
            Type::Int8 => PredictState::forward::<q8, f32>(&mut workspace, features_list),
            Type::Half => PredictState::forward::<f16, f16>(&mut workspace, features_list),
            Type::Single => PredictState::forward::<f32, f32>(&mut workspace, features_list)
        };

        drop(workspace);

        // send out our predictions to all of the receivers
        let response_iter = value_list.into_iter().zip(policy_list.into_iter());

        for (sender, response) in sender_list.into_iter().zip(response_iter) {
            sender.send(Some(response));
        }

        // wake up all of the receivers waiting for something to change
        let mut shared = self.shared.lock().unwrap();
        let num_waiting = shared.waiting_list.len();

        for waiting in shared.waiting_list.drain(0..num_waiting) {
            waiting.send(None);
        }

        // decrease the number of running neural network evaluations
        self.running_count.fetch_sub(1, Ordering::AcqRel);
    }

    fn check(&self, mut shared: MutexGuard<PredictShared>, has_more: bool) {
        let num_requests = shared.features_list.len();
        let batch_size = *config::BATCH_SIZE;

        if num_requests >= batch_size {
            // the batch is full, start an evaluation
            self.running_count.fetch_add(1, Ordering::SeqCst);
            self.predict(shared, batch_size);
        } else if has_more {
            // wait for the rest of the enqueued requests before evaluating
            // the batch
        } else if num_requests > 0 && self.running_count.compare_and_swap(0, 1, Ordering::SeqCst) == 0 {
            // nothing is running at the moment, may as well make use of
            // the device so start evaluating a partial batch
            self.predict(shared, num_requests);
        } else if num_requests > 0 && !has_more {
            // immediately evaluate when we hit a barrier in order to:
            //   1. minimize the latency between request and response
            //   2. avoid a race condition where a request that arrives
            //      during an evaluation does not trigger one.
            self.running_count.fetch_add(1, Ordering::SeqCst);
            self.predict(shared, num_requests);
        } else if num_requests == 0 && self.running_count.load(Ordering::Acquire) == 0 {
            // everything is asleep? probably a race condition between the
            // pending message being sent and it being received. Just wake
            // everything up and it should normalize.
            let num_waiting = shared.waiting_list.len();

            for waiting in shared.waiting_list.drain(0..num_waiting) {
                waiting.send(None);
            }
        }
    }
}

impl parallel::ServiceImpl for PredictState {
    type State = PredictState;
    type Request = PredictRequest;
    type Response = Option<(Singleton, Array)>;

    fn get_thread_count() -> usize {
        ::std::cmp::max(2, *config::NUM_THREADS / *config::BATCH_SIZE)
    }

    fn process(state: &Self::State, req: Self::Request, sender: OneSender<Self::Response>, has_more: bool) {
        let mut shared = state.shared.lock().unwrap();

        match req {
            PredictRequest::Ask(features) => {
                shared.features_list.push(features);
                shared.sender_list.push(sender);
            },
            PredictRequest::Wait => {
                shared.waiting_list.push(sender);
            }
        };

        state.check(shared, has_more);
    }
}
