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

use crossbeam_channel::Sender;
use std::sync::{Mutex, MutexGuard};

use super::parallel;
use super::predict::Predictor;
use dg_cuda::Device;
use dg_go::utils::features::{FEATURE_SIZE};
use dg_nn::{self as nn, Network, Workspace};
use dg_utils::types::f16;
use dg_utils::config;

pub type PredictGuard<'a> = parallel::ServiceGuard<'a, PredictState>;
pub type PredictService = parallel::Service<PredictState>;

pub fn service(network: Network) -> PredictService {
    PredictService::new(None, PredictState::new(network, 1))
}

pub enum PredictRequest {
    /// Request to compute the value and policy for some feature.
    Ask(Vec<f16>),

    /// Indicate that a worker is waiting for some other thread to finish
    /// and should be awaken after the next batch of computations finish.
    Wait,

    /// Indicate that all workers that are currently asleep should be
    /// awoken.
    Wake,
}

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

    pub fn policy(&self) -> Vec<f32> {
        self.policy.iter().map(|&x| f32::from(x)).collect()
    }
}

pub struct PredictState {
    /// The neural network weights
    network: Network,

    /// The minimum allowed batch size
    min_batch_size: usize,

    /// The batch size to use
    batch_size: usize,

    /// The features to get the value and policy for.
    features_list: Vec<f16>,

    /// The sender to response to each of the features in `features_list`
    /// over.
    sender_list: Vec<Sender<Option<PredictResponse>>>,

    /// All threads that want to get notified when something changed.
    waiting_list: Vec<Sender<Option<PredictResponse>>>,
}

impl PredictState {
    pub fn new(network: Network, min_batch_size: usize) -> PredictState {
        debug_assert!(min_batch_size <= *config::BATCH_SIZE);

        PredictState {
            network: network,
            min_batch_size: min_batch_size,
            batch_size: *config::BATCH_SIZE,
            features_list: Vec::with_capacity(*config::BATCH_SIZE * FEATURE_SIZE),
            sender_list: Vec::with_capacity(*config::BATCH_SIZE),
            waiting_list: Vec::with_capacity(*config::BATCH_SIZE)
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
    fn forward_once(workspace: &mut Workspace, features_list: &[f16]) -> Result<Vec<PredictResponse>, nn::Error> {
        let outputs = nn::forward(workspace, features_list)?;
        let (value_list, policy_list) = outputs.unwrap();
        let policy_iter = policy_list.chunks(362).map(|p| p.to_vec());

        Ok(
            value_list
                .into_iter()
                .zip(policy_iter)
                .map(|(value, policy)| PredictResponse::new(value, policy))
                .collect()
        )
    }

    /// Run the `nn::forward` function for the given features and wrap the
    /// results into `Array` elements. This version assumes the neural network
    /// use `f32` weights.
    ///
    /// # Arguments
    ///
    /// * `network` -
    /// * `batch_size` -
    /// * `features_list` -
    ///
    fn forward(network: &Network, batch_size: usize, features_list: &[f16]) -> Result<Vec<PredictResponse>, ()> {
        let mut count = 0;

        loop {
            let result = network.get_workspace(batch_size).and_then(|mut workspace| {
                PredictState::forward_once(&mut workspace, features_list)
            });

            match result {
                Ok(content) => { return Ok(content) },
                Err(reason) => {
                    eprintln!("Encountered CUDA error, retrying {} more times -- {:?}", 2 - count, reason);

                    count += 1;
                    if count >= 3 {
                        return Err(())
                    }

                    network.synchronize();
                }
            }
        }
    }

    fn predict(
        _state: &Mutex<PredictState>,
        mut state_lock: MutexGuard<PredictState>,
        batch_size: usize
    )
    {
        let num_items = state_lock.sender_list.len();
        let split_index = num_items - batch_size;
        let features_list = state_lock.features_list.split_off(split_index * FEATURE_SIZE);
        let sender_list = state_lock.sender_list.split_off(split_index);
        let network = state_lock.network.clone();  // just a bunch of Arc<...> so cheap to clone
        drop(state_lock);

        debug_assert!(features_list.len() == batch_size * FEATURE_SIZE);
        debug_assert!(sender_list.len() == batch_size);

        // perform the neural network predictions and then inform all of
        // the receivers
        if let Ok(responses) = PredictState::forward(&network, batch_size, &features_list) {
            // send out our predictions to all of the receivers
            for (sender, response) in sender_list.into_iter().zip(responses.into_iter()) {
                sender.send(Some(response)).expect("could not send predictor response");
            }
        } else {
            for sender in sender_list.into_iter() {
                sender.send(None).expect("could not send predictor nil response");
            }
        }
    }

    fn check(
        state: &Mutex<PredictState>,
        state_lock: MutexGuard<PredictState>,
        has_more: bool
    )
    {
        let num_requests = state_lock.sender_list.len();
        let batch_size = state_lock.batch_size;

        if has_more {
            if num_requests >= batch_size {
                // the batch is full, start an evaluation
                PredictState::predict(state, state_lock, batch_size);
            } else {
                // more requests are incoming, wait for them before trying to
                // evaluate a batch
            }
        } else if num_requests >= state_lock.min_batch_size {
            assert!(num_requests <= batch_size);

            // immediately evaluate when we hit a barrier in order to:
            //   1. minimize the latency between request and response
            //   2. avoid a scenario where a request is flagged as
            //      `has_more`, but the rest of the events are `Wait`
            //      events.
            PredictState::predict(state, state_lock, num_requests);
        } else {
            // wait until the currently running request finish instead of
            // waking up any threads
        }
    }
}

impl parallel::ServiceImpl for PredictState {
    type State = PredictState;
    type Request = PredictRequest;
    type Response = Option<PredictResponse>;

    fn get_thread_count() -> usize {
        let num_devices = Device::len().expect("Could not find any compatible devices");
        let num_busy = *config::NUM_THREADS / *config::BATCH_SIZE;

        ::std::cmp::max(2 * num_devices, num_busy)
    }

    fn setup_thread(index: usize) {
        let devices = Device::all().expect("Could not find any compatible devices");
        if devices.len() == 0 {
            panic!("Could not find any compatible devices");
        }
        let device = &devices[index % devices.len()];

        device.set_current().expect("Failed to set the device for the current thread")
    }

    fn process(
        state: &Mutex<Self::State>,
        mut state_lock: MutexGuard<Self::State>,
        req: Self::Request,
        sender: Sender<Self::Response>,
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
            },
            PredictRequest::Wake => {
                for waiting in state_lock.waiting_list.drain(0..) {
                    waiting.send(None).expect("failed to send predictor wake-up signal");
                }
            }
        };

        PredictState::check(state, state_lock, has_more);
    }
}

impl<T: parallel::ServiceImpl + 'static> Clone for parallel::ServiceGuard<'_, T> {
    fn clone(&self) -> Self {
        self.clone_to_static()
    }
}

impl<T> Predictor for parallel::ServiceGuard<'_, T>
    where T: parallel::ServiceImpl<Request=PredictRequest, Response=Option<PredictResponse>> + Send + 'static
{
    fn predict(&self, features: Vec<f16>) -> Option<PredictResponse> {
        self.send(PredictRequest::Ask(features))
            .expect("predict_service could not provide a response")
    }

    fn predict_all<E: Iterator<Item=Vec<f16>>>(&self, features_list: E) -> Vec<Option<PredictResponse>> {
        self.send_all(features_list.into_iter().map(|features| {
            PredictRequest::Ask(features)
        })).expect("predict_service could not provide a response")
    }

    fn wake(&self) {
        self.send_async(PredictRequest::Wake)
            .expect("predict_service could not provide a response");
    }

    fn synchronize(&self) {
        let result = self.send(PredictRequest::Wait)
            .expect("predict_service could not provide a response");

        debug_assert!(result.is_none());
    }
}
