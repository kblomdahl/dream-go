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

use crate::{Allocator, Err, Variable, AsSlice, Io};
use super::{LayerFactory, LayerImpl};

use dg_cuda::{self as cuda, cudnn};
use dg_utils::types::f16;

use std::collections::HashMap;

#[derive(Default)]
pub struct GruFactory;

impl LayerFactory for GruFactory {
    fn build(
        &self,
        _handle: &cudnn::Handle,
        _variables: &HashMap<String, Variable>,
        _stream: &cuda::Stream
    ) -> Result<Box<dyn LayerImpl>, Err>
    {
        Ok(Box::new(Gru::new()?))
    }
}

pub struct Gru {
    weight_space: cuda::PerDevice<cuda::Ptr>,
    rnn_fwd: cuda::PerDevice<HashMap<i32, cudnn::RnnForward>>,
    dev_seq_lengths: cuda::PerDevice<HashMap<i32, cuda::Ptr>>
}

impl Gru {
    pub fn new() -> Result<Self, Err> {
        Ok(Self {
            weight_space: cuda::PerDevice::new()?,
            rnn_fwd: cuda::PerDevice::new()?,
            dev_seq_lengths: cuda::PerDevice::new()?
        })
    }

    fn create_dev_seq_lengths(batch_size: i32, stream: &cuda::Stream) -> Result<cuda::Ptr, Err> {
        Ok(cuda::Ptr::from_slice(&vec! [1; batch_size as usize], stream)?)
    }

    fn create_rnn_forward(
        handle: &cudnn::Handle,
        batch_size: i32,
        units: i32
    ) -> Result<cudnn::RnnForward, Err>
    {
        Ok(cudnn::RnnForward::new(
            Self::create_rnn_descriptor(handle, units)?,
            Self::create_data_descriptor(batch_size, units)?,
            Self::create_data_descriptor(batch_size, units)?,
            Self::create_tensor_descriptor(batch_size, units)?,
        )?)
    }

    fn create_rnn_descriptor(
        handle: &cudnn::Handle,
        units: i32
    ) -> Result<cudnn::RnnDescriptor, Err>
    {
        Ok(cudnn::RnnDescriptor::new(
            &handle,
            cudnn::RnnAlgo::Standard,
            cudnn::RnnMode::Gru,
            cudnn::RnnBiasMode::DoubleBias,
            cudnn::RnnInputMode::LinearInput,
            cudnn::DataType::Half,
            cudnn::DataType::Float,
            cudnn::MathType::TensorOpMath,
            units,
            units,
            units,
            1
        )?)
    }

    fn create_data_descriptor(batch_size: i32, units: i32) -> Result<cudnn::RnnDataDescriptor, Err> {
        Ok(cudnn::RnnDataDescriptor::new(
            cudnn::DataType::Half,
            cudnn::RnnDataLayout::SeqMajorPacked,
            1,
            batch_size,
            units,
            &vec ![1; batch_size as usize]
        )?)
    }

    fn create_tensor_descriptor(batch_size: i32, units: i32) -> Result<cudnn::TensorDescriptor, Err> {
        Ok(cudnn::TensorDescriptor::new3(
            cudnn::DataType::Half,
            [1, batch_size, units],
            [batch_size * units, units, 1]
        )?)
    }
}

impl LayerImpl for Gru {
    fn build(
        &mut self,
        handle: &cudnn::Handle,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        let units: &[i32] = variables.get("units").ok_or_else(|| Err::MissingVariable("units".to_string()))?.as_slice()?;
        let rnn_desc = Self::create_rnn_descriptor(handle, units[0])?;
        let kernel = [
            variables.get("reset").ok_or_else(|| Err::MissingVariable("reset".to_string()))?,
            variables.get("update").ok_or_else(|| Err::MissingVariable("update".to_string()))?,
            variables.get("candidate").ok_or_else(|| Err::MissingVariable("candidate".to_string()))?,
            variables.get("recurrent_reset").ok_or_else(|| Err::MissingVariable("recurrent_reset".to_string()))?,
            variables.get("recurrent_update").ok_or_else(|| Err::MissingVariable("recurrent_update".to_string()))?,
            variables.get("recurrent_candidate").ok_or_else(|| Err::MissingVariable("recurrent_candidate".to_string()))?,
        ];
        let offset = [
            variables.get("reset/offset").ok_or_else(|| Err::MissingVariable("reset/offset".to_string()))?,
            variables.get("update/offset").ok_or_else(|| Err::MissingVariable("update/offset".to_string()))?,
            variables.get("candidate/offset").ok_or_else(|| Err::MissingVariable("candidate/offset".to_string()))?,
            variables.get("recurrent_reset/offset").ok_or_else(|| Err::MissingVariable("recurrent_reset/offset".to_string()))?,
            variables.get("recurrent_update/offset").ok_or_else(|| Err::MissingVariable("recurrent_update/offset".to_string()))?,
            variables.get("recurrent_candidate/offset").ok_or_else(|| Err::MissingVariable("recurrent_candidate/offset".to_string()))?,
        ];

        *self.weight_space = cuda::Ptr::new(rnn_desc.weight_space_size(handle)?)?;
        for (i, weight_param) in rnn_desc.weight_params(handle, &self.weight_space)?.iter().enumerate() {
            if weight_param[0].0.shape()?[0] > 0 {
                self.weight_space.copy_from_slice_offset::<f16>(weight_param[0].1, kernel[i].as_slice()?, stream)?;
            }
            if weight_param[1].0.shape()?[0] > 0 {
                self.weight_space.copy_from_slice_offset::<f16>(weight_param[1].1, offset[i].as_slice()?, stream)?;
            }
        }

        Ok(())
    }

    fn prepare(
        &mut self,
        handle: &cudnn::Handle,
        batch_size: i32,
        variables: &HashMap<String, Variable>,
        stream: &cuda::Stream
    ) -> Result<(), Err>
    {
        let units: &[i32] = variables.get("units").ok_or_else(|| Err::MissingVariable("units".to_string()))?.as_slice()?;

        if !self.rnn_fwd.contains_key(&batch_size) {
            self.rnn_fwd.insert(batch_size, Self::create_rnn_forward(handle, batch_size, units[0])?);
            self.dev_seq_lengths.insert(batch_size, Self::create_dev_seq_lengths(batch_size, stream)?);
        }

        Ok(())
    }

    fn forward(
        &self,
        handle: &cudnn::Handle,
        inputs: Io,
        allocator: &mut Allocator,
        _stream: &cuda::Stream,
    ) -> Result<Io, Err>
    {
        let rnn_fwd = &self.rnn_fwd[&(inputs.batch_size as i32)];
        let dev_seq_lengths = &self.dev_seq_lengths[&(inputs.batch_size as i32)];
        let output = cuda::malloc(rnn_fwd.y().size_in_bytes()?, allocator)?;
        let hidden_states = cuda::malloc(rnn_fwd.h().size_in_bytes()?, allocator)?;
        let workspace = cuda::malloc(rnn_fwd.workspace_size_in_bytes(handle)?, allocator)?;

        rnn_fwd.forward(
            handle,
            dev_seq_lengths.as_ptr() as *const _,
            inputs.current().as_ptr(),
            output.as_ptr(),
            inputs.hidden_states.as_ptr(),
            hidden_states.as_ptr(),
            self.weight_space.size_in_bytes(),
            self.weight_space.as_ptr(),
            workspace.size_in_bytes(),
            workspace.as_ptr(),
        )?;

        Ok(inputs.with_intermediate(output).with_hidden_states(hidden_states))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{Config, ExecutionPlan};
    use dg_utils::types::f16;

    fn eyes2<T: From<f32>>(n: usize) -> Vec<T> {
        let mut out = vec! [];

        for i in 0..n {
            for j in 0..n {
                out.push(T::from(if i == j { 1.0 } else { 0.0 }));
            }
        }

        out
    }

    fn zeros1<T: Clone + From<f32>>(n: usize) -> Vec<T> {
        vec! [T::from(0.0); n]
    }

    fn tanh(x: f32) -> f32 {
        x.tanh()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn random1<T: From<f32>>(n: usize, rotate: usize) -> Vec<T> {
        let mut out = vec! [];

        for i in 0..n {
            out.push(T::from(((i + rotate + 1) % n) as f32 / n as f32));
        }

        out
    }

    #[test]
    fn check_all_matrices_identity() -> Result<(), Err> {
        let n = 722;
        let factory = GruFactory::default();
        let config = Config::default().with_embeddings_size(n);
        let mut plan = ExecutionPlan::new(&cuda::Concurrent::<cuda::Sticky<cuda::Native>>::default())?;
        let hidden_states = random1::<f16>(n, 0);
        let intermediate = random1::<f16>(n, n / 2);
        let mut intermediate_ptr = cuda::malloc(2 * n, &mut plan.allocator)?;
        intermediate_ptr.copy_from_slice(&intermediate, &plan.stream)?;
        let mut io = Io::new(1, &config, &mut plan.allocator)?
            .copy_hidden_states_from(&hidden_states, &plan)?
            .with_intermediate(intermediate_ptr);
        let variables = HashMap::from([
            ("units".to_string(), Variable::from(vec! [n as i32])),
            ("reset".to_string(), Variable::from(eyes2::<f16>(n))),
            ("reset/offset".to_string(), Variable::from(zeros1::<f16>(n))),
            ("update".to_string(), Variable::from(eyes2::<f16>(n))),
            ("update/offset".to_string(), Variable::from(zeros1::<f16>(n))),
            ("candidate".to_string(), Variable::from(eyes2::<f16>(n))),
            ("candidate/offset".to_string(), Variable::from(zeros1::<f16>(n))),
            ("recurrent_reset".to_string(), Variable::from(eyes2::<f16>(n))),
            ("recurrent_reset/offset".to_string(), Variable::from(zeros1::<f16>(n))),
            ("recurrent_update".to_string(), Variable::from(eyes2::<f16>(n))),
            ("recurrent_update/offset".to_string(), Variable::from(zeros1::<f16>(n))),
            ("recurrent_candidate".to_string(), Variable::from(eyes2::<f16>(n))),
            ("recurrent_candidate/offset".to_string(), Variable::from(zeros1::<f16>(n))),
        ]);
        let mut layer = factory.build(&plan.handle, &variables, &plan.stream)?;
        layer.build(&plan.handle, &variables, &plan.stream)?;
        layer.prepare(&plan.handle, 1, &variables, &plan.stream)?;
        io = layer.forward(&plan.handle, io, &mut plan.allocator, &plan.stream)?;
        let out_intermediate: Vec<f16> = io.intermediate.to_vec(&plan.stream)?;
        let out_hidden_states: Vec<f16> = io.hidden_states.to_vec(&plan.stream)?;

        for i in 0..n {
            let x = f32::from(intermediate[i]);
            let h = f32::from(hidden_states[i]);
            let z = sigmoid(x + h);
            let r = sigmoid(x + h);
            let h_hat = tanh(x + r * h);
            let h_1 = (1.0 - z) * h_hat + z * h;

            assert!(
                f32::from(out_intermediate[i]) >= h_1 - 1e-3 &&
                f32::from(out_intermediate[i]) <= h_1 + 1e-3,
                "intermediate[{}]: {:?} is not almost equal to {:?}, delta is {}",
                i,
                out_intermediate[i],
                h_1,
                (f32::from(out_intermediate[i]) - h_1).abs()
            );

            assert!(
                f32::from(out_hidden_states[i]) >= h_1 - 1e-3 &&
                f32::from(out_hidden_states[i]) <= h_1 + 1e-3,
                "hidden_states[{}]: {:?} is not almost equal to {:?}, delta is {}",
                i,
                out_hidden_states[i],
                h_1,
                (f32::from(out_hidden_states[i]) - h_1).abs()
            );
        }

        Ok(())
    }
}
