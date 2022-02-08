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

use dg_cuda::{self as cuda, cudnn, cublas_lt};
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

        for device in cuda::Device::all()? {
            device.set_current()?;
            let handle = cudnn::Handle::new()?;
            let light_handle = cublas_lt::Handle::new()?;
            let stream = cuda::Stream::new()?;

            Self::build_layers(&mut self.representation, &light_handle, &handle, &stream)?;
            Self::build_layers(&mut self.dynamics, &light_handle, &handle, &stream)?;
            Self::build_layers(&mut self.gru, &light_handle, &handle, &stream)?;
            Self::build_layers(&mut self.prediction, &light_handle, &handle, &stream)?;

            streams.push(stream);
        }

        for stream in streams.drain(..) {
            stream.synchronize()?;
        }

        Ok(())
    }

    fn build_layers(layers: &mut [Layer], light_handle: &cublas_lt::Handle, handle: &cudnn::Handle, stream: &cuda::Stream) -> Result<(), Err> {
        for layer in layers {
            layer.build(light_handle, handle, stream)?;
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
            io = layer.forward(&plan.light_handle, &plan.handle, io, &mut plan.allocator, &plan.stream)?;
        }
        io = io.with_intermediate_as_hidden_states(&plan.stream)?;

        for layer in &self.prediction {
            io = layer.forward(&plan.light_handle, &plan.handle, io, &mut plan.allocator, &plan.stream)?;
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
            io = layer.forward(&plan.light_handle, &plan.handle, io, &mut plan.allocator, &plan.stream)?;
        }

        plan.stream.wait_event(&plan.hidden_states_copy_finished)?;
        for layer in &self.gru {
            io = layer.forward(&plan.light_handle, &plan.handle, io, &mut plan.allocator, &plan.stream)?;
        }

        for layer in &self.prediction {
            io = layer.forward(&plan.light_handle, &plan.handle, io, &mut plan.allocator, &plan.stream)?;
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
    use test::Bencher;
    use std::fs::File;

    fn check_array_approx_eq<T>(a: &[T], b: &[T], eps: f32)
        where f32: From<T>, T: Copy
    {
        assert_eq!(a.len(), b.len());

        for i in 0..a.len() {
            let a = f32::from(a[i]);
            let b = f32::from(b[i]);

            assert!(
                a >= b - eps && a <= b + eps,
                "[{}]: {} is not almost equal to {}, delta is {}",
                i,
                a,
                b,
                (a - b).abs()
            );
        }
    }

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

            // representation
            let output = model.initial_predict(&fake_features, 1).expect("could not predict model output");
            let test_policy: &[f16] = tests["p1"].as_slice().expect("could not retrieve policy test values");
            let test_value: &[f16] = tests["v1"].as_slice().expect("could not retrieve value test values");

            check_array_approx_eq(&output.policy[0..362], &test_policy[0..362], 3e-4);
            check_array_approx_eq(&output.value[0..1], &test_value[0..1], 2e-3);

            // representation -> dynamics
            let output = model.predict(&output.hidden_states, &fake_features, 1).expect("could not predict model output");
            let test_policy: &[f16] = tests["p2"].as_slice().expect("could not retrieve policy test values");
            let test_value: &[f16] = tests["v2"].as_slice().expect("could not retrieve value test values");

            check_array_approx_eq(&output.policy[0..362], &test_policy[0..362], 5e-4);
            check_array_approx_eq(&output.value[0..1], &test_value[0..1], 2e-3);

            // dynamics -> dynamics
            let output = model.predict(&output.hidden_states, &fake_features, 1).expect("could not predict model output");
            let test_policy: &[f16] = tests["p3"].as_slice().expect("could not retrieve policy test values");
            let test_value: &[f16] = tests["v3"].as_slice().expect("could not retrieve value test values");

            check_array_approx_eq(&output.policy[0..362], &test_policy[0..362], 5e-4);
            check_array_approx_eq(&output.value[0..1], &test_value[0..1], 2e-3);
        }
    }

    #[bench]
    fn initial_predict_01(b: &mut Bencher) {
        let root = env!("CARGO_MANIFEST_DIR");
        let file = File::open(&format!("{}/../../dream_go.json", root));
        if let Ok(file) = file {
            let model = Builder::parse(file)
                .expect("could not parse model")
                .build()
                .expect("could not build model");
            let num_features = model.config.num_features;
            let fake_features = vec! [f16::from(1.0); 19 * 19 * num_features];

            b.iter(|| {
                model.initial_predict(&fake_features, 1).expect("could not predict model output")
            });
        }
    }

    #[bench]
    fn predict_01(b: &mut Bencher) {
        let root = env!("CARGO_MANIFEST_DIR");
        let file = File::open(&format!("{}/../../dream_go.json", root));
        if let Ok(file) = file {
            let model = Builder::parse(file)
                .expect("could not parse model")
                .build()
                .expect("could not build model");
            let num_features = model.config.num_features;
            let embeddings_size = model.config.embeddings_size;
            let fake_features = vec! [f16::from(1.0); 19 * 19 * num_features];
            let fake_hidden_states = vec! [f16::from(1.0); embeddings_size];

            b.iter(|| {
                model.predict(&fake_hidden_states, &fake_features, 1).expect("could not predict model output")
            });
        }
    }
}
