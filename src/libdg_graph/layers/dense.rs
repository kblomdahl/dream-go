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

use std::sync::Arc;

use libc::c_void;

use ::graph_def::{ActivationTypeDef, LayerDef};
use ::layer::{Layer, PreparedLayer};
use dg_cuda as cuda;
use dg_cuda::cudnn;
use factories::{activation_factory, convolution_factory, filter_factory, tensor_factory};

#[derive(Debug)]
pub struct Dense {
    bias: Arc<cudnn::Tensor>,
    bias_data: Arc<cuda::Ptr>,
    filter: Arc<cudnn::Filter>,
    filter_data: Arc<cuda::Ptr>,
    activation_1: Arc<cudnn::Activation>,
    activation_2: Option<Arc<cudnn::Activation>>,
    convolution: Arc<cudnn::Convolution>
}

struct PreparedDense {
    bias: Arc<cudnn::Tensor>,
    bias_data: Arc<cuda::Ptr>,
    filter: Arc<cudnn::Filter>,
    filter_data: Arc<cuda::Ptr>,
    convolution: Arc<cudnn::Convolution>,
    activation_1: Arc<cudnn::Activation>,
    activation_2: Option<Arc<cudnn::Activation>>,
    fwd_algo: cudnn::cudnnConvolutionFwdAlgoPerf_t
}

impl Layer for Dense {
    fn prepare(
        &self,
        handle: &cudnn::Handle,
        inputs: &[&cudnn::Tensor],
        outputs: &[&cudnn::Tensor]
    ) -> Result<Box<PreparedLayer>, cuda::Error>
    {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let fwd_algo = self.convolution.compile_for(
            handle,
            inputs[0],
            &self.filter,
            outputs[0],
            cudnn::cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        )?;

        Ok(Box::new(PreparedDense {
            bias: self.bias.clone(),
            bias_data: self.bias_data.clone(),
            filter: self.filter.clone(),
            filter_data: self.filter_data.clone(),
            convolution: self.convolution.clone(),
            activation_1: self.activation_1.clone(),
            activation_2: self.activation_2.clone(),
            fwd_algo
        }))
    }
}

impl PreparedLayer for PreparedDense {
    fn size_in_bytes(&self) -> usize {
        self.fwd_algo.memory
    }

    fn forward(
        &self,
        handle: &cudnn::Handle,
        inputs: &[(&cudnn::Tensor, *const c_void)],
        outputs: &[(&cudnn::Tensor, *mut c_void)],
        workspace_ptr: *mut c_void
    ) -> Result<(), cuda::Error>
    {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        self.convolution.forward(
            handle,
            inputs[0].0, inputs[0].1,
            &self.filter, self.filter_data.as_ptr(),
            &self.fwd_algo, workspace_ptr,
            &self.bias, self.bias_data.as_ptr(),
            &self.activation_1,
            outputs[0].0, outputs[0].1
        )?;

        if let Some(ref act) = self.activation_2 {
            act.forward(
                handle,
                outputs[0].0, outputs[0].1,
                outputs[0].0, outputs[0].1
            )
        } else {
            Ok(())
        }
    }
}

impl Dense {
    pub fn new(layer_def: &LayerDef) -> Result<Dense, cuda::Error> {
        assert!(layer_def.arguments.is_some());

        let arguments = layer_def.arguments.as_ref().unwrap();
        let mode = match arguments.activation {
            ActivationTypeDef::ReLU => cudnn::cudnnActivationMode_t::ReLU,
            ActivationTypeDef::Sigmoid => cudnn::cudnnActivationMode_t::Identity,
            ActivationTypeDef::Linear => cudnn::cudnnActivationMode_t::Identity,
        };

        Self::with_activation(
            layer_def,
            arguments.activation,
            activation_factory::get_or_create(
                mode,
                cudnn::cudnnNanPropagation_t::NotPropagateNaN,
            )?
        )
    }

    fn with_activation(
        layer_def: &LayerDef,
        activation_type: ActivationTypeDef,
        activation_1: Arc<cudnn::Activation>
    ) -> Result<Dense, cuda::Error>
    {
        let arguments = layer_def.arguments.as_ref().unwrap();
        let arg_kernel = arguments.kernel.as_ref().unwrap();
        let arg_bias = arguments.bias.as_ref().unwrap();

        // use cuDNN to perform the matrix multiplication, by modelling our
        // matrix multiplication as a 1x1 convolution (*). This allows us to
        // avoid the cuBLAS dependency.
        //
        // (*) going the full circle, since cuDNN model convolutions as
        //     matrix multiplication.
        //
        let filter = filter_factory::get_or_create(
            cudnn::cudnnDataType_t::Half,
            cudnn::cudnnTensorFormat_t::NCHW,
            &vec! [arg_kernel.shape[0] as usize, arg_kernel.shape[1] as usize, 1, 1]
        )?;
        let convolution = convolution_factory::get_or_create(&filter, 1)?;
        let bias_size = arg_bias.shape[0] as usize;
        let bias = tensor_factory::get_or_create(
            cudnn::cudnnDataType_t::Half,
            cudnn::cudnnTensorFormat_t::NCHW,
            &[1, bias_size, 1, 1]
        )?;

        Ok(Dense {
            bias,
            bias_data: Arc::new(cuda::Ptr::from_vec(&arg_bias.value, &cuda::Stream::default())?),
            filter,
            filter_data: Arc::new(cuda::Ptr::from_vec(&arg_kernel.value, &cuda::Stream::default())?),
            activation_1,
            activation_2: match activation_type {
                ActivationTypeDef::ReLU | ActivationTypeDef::Linear => None,
                ActivationTypeDef::Sigmoid => {
                    Some(activation_factory::get_or_create(
                        cudnn::cudnnActivationMode_t::Sigmoid,
                        cudnn::cudnnNanPropagation_t::NotPropagateNaN
                    )?)
                }
            },
            convolution
        })
    }
}

#[cfg(test)]
mod tests {
    use dg_cuda::cudnn::cudnnDataType_t;
    use dg_utils::types::f16;
    use graph_def::{ActivationTypeDef, ConstantDef, ConstantValueDef, LayerArgumentsDef, LayerTypeDef, VariableDef};
    use layers::tests::{assert_approx_eq_prec, run_layer};

    use super::*;
    use rand::{thread_rng, Rng};
    use std::f32::consts::E;

    fn matmul(a: &[f32], b: &[f32], bias: &[f32], n: usize, m: usize, p: usize) -> Vec<f32> {
        let mut out = vec! [0.0f32; n*p];

        for i in 0..n {
            for j in 0..m {
                for k in 0..p {
                    out[i*p+k] += a[i*m+j] * b[k*m+j];
                }
            }
        }

        for i in 0..n {
            for k in 0..p {
                out[i*p+k] += bias[k];
            }
        }

        out
    }

    fn with_activation<F: Fn(f32) -> f32>(activation: ActivationTypeDef, act: F) {
        let kernel: Vec<_> = (0..(362*16))
            .map(|_i| 1.0 - 2.0 * thread_rng().gen::<f32>())
            .collect();
        let bias: Vec<_> = (0..16)
            .map(|_i| 5.0 - 10.0 * thread_rng().gen::<f32>())
            .collect();

        let layer_def = LayerDef {
            type_of: LayerTypeDef::Dense,
            input: vec! [
                VariableDef { id: 0, shape: vec! [4, 1, 1, 362] }
            ],
            output: vec! [
                VariableDef { id: 1, shape: vec! [4, 1, 1, 16] }
            ],
            arguments: Some(LayerArgumentsDef {
                kernel: Some(ConstantDef {
                    shape: vec! [16, 362],
                    value: ConstantValueDef {
                        inner: Arc::new(kernel.iter().map(|&x| f16::from(x)).collect())
                    }
                }),
                bias: Some(ConstantDef {
                    shape: vec! [16],
                    value: ConstantValueDef {
                        inner: Arc::new(bias.iter().map(|&x| f16::from(x)).collect())
                    }
                }),
                alpha: None,
                group_count: 1,
                activation
            })
        };
        let layer = Dense::new(&layer_def)
            .expect("Could not create dense layer");

        let (inputs, outputs) = run_layer::<f16, _>(
            &layer_def,
            &layer,
            cudnnDataType_t::Half
        );

        let input: Vec<f32> = inputs[0].iter().map(|&x| f32::from(x)).collect();
        let expected_output = matmul(&input, &kernel, &bias, 4, 362, 16);

        for (expected_outp, &outp) in expected_output.into_iter().zip(outputs[0].iter()) {
            let expected_outp = act(expected_outp);

            assert_approx_eq_prec(f32::from(outp), expected_outp, 0.01);
        }
    }

    #[test]
    fn dense_relu() {
        with_activation(
            ActivationTypeDef::ReLU,
            |x| if x > 0.0 { x } else { 0.0 }
        )
    }

    #[test]
    fn dense_sigmoid() {
        with_activation(
            ActivationTypeDef::Sigmoid,
            |x| 1.0 / (1.0 + E.powf(-x))
        )
    }

    #[test]
    fn dense_linear() {
        with_activation(
            ActivationTypeDef::Linear,
            |x| x
        )
    }
}