// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use crate::{Config, Err, ExecutionPlan, Io, Output};
use super::Layer;

use dg_cuda as cuda;
use dg_cuda::cudnn;
use dg_utils::types::f16;

pub struct Model {
    allocator: cuda::PerDevice<cuda::Concurrent<cuda::Sticky<cuda::Native>>>,
    config: Config,

    representation: Vec<Layer>,
    dynamics: Vec<Layer>,
    gru: Vec<Layer>,
    prediction: Vec<Layer>
}

impl Model {
    pub fn new(
        config: Config,
        representation: Vec<Layer>,
        dynamics: Vec<Layer>,
        gru: Vec<Layer>,
        prediction: Vec<Layer>
    ) -> Result<Self, Err>
    {
        let allocator = cuda::PerDevice::new()?;
        let mut out = Self {
            allocator,
            config,
            representation,
            dynamics,
            gru,
            prediction
        };

        out.build()?;
        Ok(out)
    }

    fn build(&mut self) -> Result<(), Err> {
        let mut streams = vec! [];
        let handle = cudnn::Handle::new()?;

        for device in cuda::Device::all()? {
            device.set_current()?;

            let stream = cuda::Stream::new()?;

            Self::build_layers(&mut self.representation, &handle, &stream)?;
            Self::build_layers(&mut self.dynamics, &handle, &stream)?;
            Self::build_layers(&mut self.gru, &handle, &stream)?;
            Self::build_layers(&mut self.prediction, &handle, &stream)?;

            streams.push(stream);
        }

        for stream in streams.drain(..) {
            stream.synchronize()?;
        }

        Ok(())
    }

    fn build_layers(layers: &mut [Layer], handle: &cudnn::Handle, stream: &cuda::Stream) -> Result<(), Err> {
        for layer in layers {
            layer.build(handle, stream)?;
        }

        Ok(())
    }

    pub fn initial_predict(
        &self,
        features: &[f16],
        batch_size: usize
    ) -> Result<Output, Err>
    {
        let mut plan = self.get_available_execution_plan()?;
        let mut io = Io::new(batch_size, &self.config, &mut plan.allocator)?
            .copy_features_from(features, &plan)?;

        plan.stream.wait_event(&plan.features_copy_finished)?;
        for layer in &self.representation {
            io = layer.forward(&plan.handle, io, &mut plan.allocator, &plan.stream)?;
        }
        io = io.with_intermediate_as_hidden_states(&plan.stream)?;

        for layer in &self.prediction {
            io = layer.forward(&plan.handle, io, &mut plan.allocator, &plan.stream)?;
        }

        Ok(io.as_outputs(&plan.stream)?)
    }

    pub fn predict(
        &self,
        hidden_states: &[f16],
        features: &[f16],
        batch_size: usize
    ) -> Result<Output, Err>
    {
        let mut plan = self.get_available_execution_plan()?;
        let mut io = Io::new(batch_size, &self.config, &mut plan.allocator)?
            .copy_features_from(features, &plan)?
            .copy_hidden_states_from(hidden_states, &plan)?;

        plan.stream.wait_event(&plan.features_copy_finished)?;
        for layer in &self.dynamics {
            io = layer.forward(&plan.handle, io, &mut plan.allocator, &plan.stream)?;
        }

        plan.stream.wait_event(&plan.hidden_states_copy_finished)?;
        for layer in &self.gru {
            io = layer.forward(&plan.handle, io, &mut plan.allocator, &plan.stream)?;
        }

        for layer in &self.prediction {
            io = layer.forward(&plan.handle, io, &mut plan.allocator, &plan.stream)?;
        }

        Ok(io.as_outputs(&plan.stream)?)
    }

    fn get_available_execution_plan(&self) -> Result<ExecutionPlan, Err> {
        ExecutionPlan::new(&self.allocator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{builder::Builder, AsSlice};
    use std::fs::File;

    #[test]
    fn check_initial_predict() {
        let root = env!("CARGO_MANIFEST_DIR");
        let file = File::open(&format!("{}/../../dream_go.json", root));
        if let Ok(file) = file {
            let (model, tests) = Builder::parse(file)
                .expect("could not parse model")
                .build_with_tests()
                .expect("could not build model");
            let num_features = model.config.num_features;
            let fake_features = vec! [f16::from(1.0); 19 * 19 * num_features];
            let output = model.initial_predict(&fake_features, 1).expect("could not predict model output");
            let test_policy: &[f16] = tests["p1"].as_slice().expect("could not retrieve policy test values");
            let test_value: &[f16] = tests["v1"].as_slice().expect("could not retrieve value test values");

            for i in 0..362 {
                assert!(
                    f32::from(output.policy[i]) >= f32::from(test_policy[i]) - 1e-4 &&
                    f32::from(output.policy[i]) <= f32::from(test_policy[i]) + 1e-4,
                    "policy[{}]: {:?} is not almost equal to {:?}, delta is {}",
                    i,
                    output.policy[i],
                    test_policy[i],
                    (f32::from(output.policy[i]) - f32::from(test_policy[i])).abs()
                );
            }

            assert!(
                f32::from(output.value[0]) >= f32::from(test_value[0]) - 1e-3 &&
                f32::from(output.value[0]) <= f32::from(test_value[0]) + 1e-3,
                "value: {:?} is not almost equal to {:?}, delta is {}",
                output.value[0],
                test_value[0],
                (f32::from(output.value[0]) - f32::from(test_value[0])).abs()
            );

            let output = model.predict(&output.hidden_states, &fake_features, 1).expect("could not predict model output");
            let test_policy: &[f16] = tests["p2"].as_slice().expect("could not retrieve policy test values");
            let test_value: &[f16] = tests["v2"].as_slice().expect("could not retrieve value test values");

            assert!(
                f32::from(output.value[0]) >= f32::from(test_value[0]) - 1e-3 &&
                f32::from(output.value[0]) <= f32::from(test_value[0]) + 1e-3,
                "value: {:?} is not almost equal to {:?}, delta is {}",
                output.value[0],
                test_value[0],
                (f32::from(output.value[0]) - f32::from(test_value[0])).abs()
            );

            for i in 0..362 {
                assert!(
                    f32::from(output.policy[i]) >= f32::from(test_policy[i]) - 2e-4 &&
                    f32::from(output.policy[i]) <= f32::from(test_policy[i]) + 2e-4,
                    "policy[{}]: {:?} is not almost equal to {:?}, delta is {}",
                    i,
                    output.policy[i],
                    test_policy[i],
                    (f32::from(output.policy[i]) - f32::from(test_policy[i])).abs()
                );
            }
        }
    }
}
