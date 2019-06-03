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

use libc::c_void;

use ::graph_def::{ActivationTypeDef, LayerDef};
use ::layer::{Layer, PreparedLayer};
use dg_cuda as cuda;
use dg_cuda::cudnn;
use factories::{tensor_factory, filter_factory, activation_factory, convolution_factory};
use std::sync::Arc;

#[derive(Debug)]
pub struct Conv2D {
    bias: Arc<cudnn::Tensor>,
    bias_data: Arc<cuda::Ptr>,
    filter: Arc<cudnn::Filter>,
    filter_data: Arc<cuda::Ptr>,
    convolution: Arc<cudnn::Convolution>,
    activation: Arc<cudnn::Activation>,
}

struct PreparedConv2D {
    bias: Arc<cudnn::Tensor>,
    bias_data: Arc<cuda::Ptr>,
    filter: Arc<cudnn::Filter>,
    filter_data: Arc<cuda::Ptr>,
    convolution: Arc<cudnn::Convolution>,
    activation: Arc<cudnn::Activation>,
    fwd_algo: cudnn::cudnnConvolutionFwdAlgoPerf_t
}

impl Layer for Conv2D {
    fn prepare(
        &self,
        handle: &cudnn::Handle,
        inputs: &[&cudnn::Tensor],
        outputs: &[&cudnn::Tensor]
    ) -> Result<Box<PreparedLayer>, cuda::Error>
    {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let fwd_algo = self.convolution.compile(
            handle,
            inputs[0],
            &self.filter,
            outputs[0],
            &self.activation
        )?;

        Ok(Box::new(PreparedConv2D {
            bias: self.bias.clone(),
            bias_data: self.bias_data.clone(),
            filter: self.filter.clone(),
            filter_data: self.filter_data.clone(),
            convolution: self.convolution.clone(),
            activation: self.activation.clone(),
            fwd_algo
        }))
    }
}

impl PreparedLayer for PreparedConv2D {
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
            &self.activation,
            outputs[0].0, outputs[0].1
        )
    }
}

impl Conv2D {
    pub fn new(layer_def: &LayerDef) -> Result<Conv2D, cuda::Error> {
        assert!(layer_def.arguments.is_some());

        let arguments = layer_def.arguments.as_ref().unwrap();
        let mode = match arguments.activation {
            ActivationTypeDef::ReLU => cudnn::cudnnActivationMode_t::ReLU,
            ActivationTypeDef::Sigmoid => cudnn::cudnnActivationMode_t::Sigmoid,
            ActivationTypeDef::Linear => cudnn::cudnnActivationMode_t::Identity,
        };

        Self::with_activation(
            layer_def,
            activation_factory::get_or_create(mode, cudnn::cudnnNanPropagation_t::NotPropagateNaN)?
        )
    }

    fn with_activation(layer_def: &LayerDef, activation: Arc<cudnn::Activation>) -> Result<Conv2D, cuda::Error> {
        let arguments = layer_def.arguments.as_ref().unwrap();
        let arg_group_count = arguments.group_count;
        let arg_kernel = arguments.kernel.as_ref().unwrap();
        let arg_bias = arguments.bias.as_ref().unwrap();

        let filter_dims: Vec<usize> = arg_kernel.shape.iter()
            .map(|&x| x as usize)
            .collect();
        let is_1x1 = filter_dims[1] == 1 && filter_dims[2] == 1;
        let filter = filter_factory::get_or_create(
            cudnn::cudnnDataType_t::Half,
            if is_1x1 { cudnn::cudnnTensorFormat_t::NCHW } else { cudnn::cudnnTensorFormat_t::NHWC },
            &vec! [filter_dims[0], filter_dims[3], filter_dims[1], filter_dims[2]]
        )?;
        let convolution = convolution_factory::get_or_create(&filter, arg_group_count)?;
        let bias_size = arg_bias.shape[0] as usize;
        let bias = tensor_factory::get_or_create(
            cudnn::cudnnDataType_t::Half,
            cudnn::cudnnTensorFormat_t::NCHW,
            &[1, bias_size, 1, 1]
        )?;

        Ok(Conv2D {
            activation,
            bias,
            bias_data: Arc::new(cuda::Ptr::from_vec(&arg_bias.value, &cuda::Stream::default())?),
            filter,
            filter_data: Arc::new(cuda::Ptr::from_vec(&arg_kernel.value, &cuda::Stream::default())?),
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

    fn conv2d(
        inp: &[f32],
        weights: &[f32],
        bias: &[f32],
        batch_size: usize,
        image_height: usize,
        image_width: usize,
        output_channels: usize,
        filter_height: usize,
        filter_width: usize,
        input_channels: usize
    ) -> Vec<f32>
    {
        let mut out = vec! [
            0.0f32;
            batch_size*image_height*image_width*output_channels
        ];

        for n in 0..batch_size {
            for p in 0..image_height {
                for q in 0..image_width {
                    for k in 0..output_channels {
                        let mut y_ = bias[k];

                        for r in 0..filter_height {
                            for s in 0..filter_width {
                                for c in 0..input_channels {
                                    if (p as isize + r as isize - 1) < 0 {
                                        continue;
                                    } else if (p as isize + r as isize - 1) >= image_height as isize {
                                        continue;
                                    } else if (q as isize + s as isize - 1) < 0 {
                                        continue;
                                    } else if (q as isize + s as isize - 1) >= image_width as isize {
                                        continue;
                                    }

                                    let x_ = inp[
                                        input_channels * image_width * image_height * n
                                        + input_channels * image_width * (p+r-1)
                                        + input_channels * (q+s-1)
                                        + c
                                    ];
                                    let w_ = weights[
                                        input_channels * filter_width * filter_height * k
                                        + input_channels * filter_width * r
                                        + input_channels * s
                                        + c
                                    ];

                                    y_ += w_ * x_;
                                }
                            }
                        }

                        out[
                            image_height*image_width*output_channels * n
                            + image_width*output_channels * p
                            + output_channels * q
                            + k
                        ] = y_;
                    }
                }
            }
        }

        out
    }

    fn with_activation<F: Fn(f32) -> f32>(activation: ActivationTypeDef, act: F) {
        let kernel: Vec<_> = (0..(32*3*3*64))
            .map(|_i| 1.0 - 2.0 * thread_rng().gen::<f32>())
            .collect();
        let bias: Vec<_> = (0..32)
            .map(|_i| 5.0 - 10.0 * thread_rng().gen::<f32>())
            .collect();

        let layer_def = LayerDef {
            type_of: LayerTypeDef::Conv2D,
            input: vec! [
                VariableDef { id: 0, shape: vec! [2, 19, 19, 64] }
            ],
            output: vec! [
                VariableDef { id: 0, shape: vec! [2, 19, 19, 32] }
            ],
            arguments: Some(LayerArgumentsDef {
                kernel: Some(ConstantDef {
                    shape: vec! [32, 3, 3, 64],
                    value: ConstantValueDef {
                        inner: Arc::new(kernel.iter().map(|&x| f16::from(x)).collect())
                    }
                }),
                bias: Some(ConstantDef {
                    shape: vec! [32],
                    value: ConstantValueDef {
                        inner: Arc::new(bias.iter().map(|&x| f16::from(x)).collect())
                    }
                }),
                alpha: None,
                group_count: 1,
                activation
            })
        };
        let layer = Conv2D::new(&layer_def)
            .expect("Could not create conv2d layer");

        let (inputs, outputs) = run_layer::<f16, _>(
            &layer_def,
            &layer,
            cudnnDataType_t::Half
        );

        let input: Vec<f32> = inputs[0].iter().map(|&x| f32::from(x)).collect();
        let expected_output = conv2d(
            &input,
            &kernel,
            &bias,
            2, 19, 19, 32, 3, 3, 64
        );

        for (expected_outp, &outp) in expected_output.into_iter().zip(outputs[0].iter()) {
            let expected_outp = act(expected_outp);

            assert_approx_eq_prec(f32::from(outp), expected_outp, 0.02);
        }
    }

    #[test]
    fn conv2d_relu() {
        with_activation(
            ActivationTypeDef::ReLU,
            |x| if x > 0.0 { x } else { 0.0 }
        )
    }

    #[test]
    fn conv2d_linear() {
        with_activation(
            ActivationTypeDef::Linear,
            |x| x
        )
    }
}