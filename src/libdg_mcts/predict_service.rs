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

use super::predict::{Predictor, PredictResponse};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use dg_cuda::Device;
use dg_nn::{self as nn, Network};
use dg_utils::types::f16;

#[derive(Clone)]
pub struct PredictService {
    network: Network,
    count: Arc<AtomicUsize>
}

impl PredictService {
    pub fn new(network: Network) -> Self {
        let count = Arc::new(AtomicUsize::new(0));

        Self { network, count }
    }
}

impl Predictor for PredictService {
    fn max_num_threads(&self) -> usize {
        let num_devices = Device::all().expect("could not find any compatible devices").len();
        2 * num_devices
    }

    fn predict(&self, features_list: &[f16], batch_size: usize) -> Vec<PredictResponse> {
        let devices = Device::all().expect("could not find any compatible devices");
        let index = self.count.fetch_add(1, Ordering::Relaxed) % devices.len();
        devices[index].set_current().expect("could not set the device for the current thread");

        //
        let network = &self.network;
        let result = network.get_workspace(batch_size).and_then(|mut workspace| {
            let outputs = nn::forward(&mut workspace, features_list)?;
            let (value_list, policy_list) = outputs.unwrap();
            let policy_iter = policy_list.chunks(362).map(|p| p.to_vec());

            Ok(
                value_list
                    .into_iter()
                    .zip(policy_iter)
                    .map(|(value, policy)| PredictResponse::new(value, policy))
                    .collect()
            )
        });

        result.expect("could not run neural network")
    }
}
