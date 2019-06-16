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
    activation_1: Arc<cudnn::Activation>,
    activation_2: Option<Arc<cudnn::Activation>>,
}

struct PreparedConv2D {
    bias: Arc<cudnn::Tensor>,
    bias_data: Arc<cuda::Ptr>,
    filter: Arc<cudnn::Filter>,
    filter_data: Arc<cuda::Ptr>,
    convolution: Arc<cudnn::Convolution>,
    activation_1: Arc<cudnn::Activation>,
    activation_2: Option<Arc<cudnn::Activation>>,
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
            &self.activation_1
        )?;

        Ok(Box::new(PreparedConv2D {
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
            &self.activation_1,
            outputs[0].0, outputs[0].1
        )?;

        if let Some(ref act) = self.activation_2 {
            act.forward(
                handle,
                outputs[0].0, outputs[0].1,
                outputs[0].0, outputs[0].1
            )?;
        }

        Ok(())
    }
}

impl Conv2D {
    pub fn new(layer_def: &LayerDef) -> Result<Conv2D, cuda::Error> {
        assert!(layer_def.arguments.is_some());

        let arguments = layer_def.arguments.as_ref().unwrap();
        let mode_1 = match arguments.activation {
            ActivationTypeDef::ReLU => cudnn::cudnnActivationMode_t::ReLU,
            ActivationTypeDef::Sigmoid => cudnn::cudnnActivationMode_t::Identity,
            ActivationTypeDef::Linear => cudnn::cudnnActivationMode_t::Identity,
        };

        Self::with_activation(
            layer_def,
            activation_factory::get_or_create(mode_1, cudnn::cudnnNanPropagation_t::NotPropagateNaN)?,
            match arguments.activation {
                ActivationTypeDef::ReLU => None,
                ActivationTypeDef::Linear => None,
                ActivationTypeDef::Sigmoid => Some(activation_factory::get_or_create(cudnn::cudnnActivationMode_t::Sigmoid, cudnn::cudnnNanPropagation_t::NotPropagateNaN)?),
            }
        )
    }

    fn with_activation(
        layer_def: &LayerDef,
        activation_1: Arc<cudnn::Activation>,
        activation_2: Option<Arc<cudnn::Activation>>
    ) -> Result<Conv2D, cuda::Error>
    {
        let arguments = layer_def.arguments.as_ref().unwrap();
        let arg_group_count = arguments.group_count;
        let arg_kernel = arguments.kernel.as_ref().unwrap();
        let arg_bias = arguments.bias.as_ref().unwrap();
        let data_type = if layer_def.is_input_half() {
            cudnn::cudnnDataType_t::Half
        } else {
            cudnn::cudnnDataType_t::Float
        };

        // construct the filter with a special case for 1x1 filters that seems to be
        // necessary to avoid cuDNN raising an UNSUPPORTED error. The workaround is to
        // use NCHW instead of NHWC even though HW = 1 so the ordering makes no
        // difference.
        let filter_dims: Vec<usize> = arg_kernel.shape.iter()
            .map(|&x| x as usize)
            .collect();
        let is_1x1 = filter_dims[1] == 1 && filter_dims[2] == 1;
        let filter = filter_factory::get_or_create(
            data_type,
            if is_1x1 { cudnn::cudnnTensorFormat_t::NCHW } else { cudnn::cudnnTensorFormat_t::NHWC },
            &vec! [filter_dims[0], filter_dims[3], filter_dims[1], filter_dims[2]]
        )?;
        let filter_data = filter.convert_to_ptr(&arg_kernel.value)?;

        // convolution and bias
        let convolution = convolution_factory::get_or_create(&filter, arg_group_count)?;
        let bias_size = arg_bias.shape[0] as usize;
        let bias = tensor_factory::get_or_create(
            data_type,
            cudnn::cudnnTensorFormat_t::NCHW,
            &[1, bias_size, 1, 1]
        )?;
        let bias_data = bias.convert_to_ptr(&arg_bias.value)?;

        Ok(Conv2D {
            activation_1,
            activation_2,
            bias,
            bias_data: Arc::new(bias_data),
            filter,
            filter_data: Arc::new(filter_data),
            convolution
        })
    }
}

#[cfg(test)]
mod tests {
    use dg_utils::types::f16;
    use graph_def::{ActivationTypeDef, ConstantDef, ConstantValueDef, LayerArgumentsDef, LayerTypeDef, VariableDef, DataTypeDef};
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

        let ph = (filter_height - 1) / 2;
        let pw = (filter_width - 1) / 2;

        for n in 0..batch_size {
            for p in 0..image_height {
                for q in 0..image_width {
                    for k in 0..output_channels {
                        let mut y_ = bias[k];

                        for r in 0..filter_height {
                            for s in 0..filter_width {
                                for c in 0..input_channels {
                                    if (p as isize + r as isize - ph as isize) < 0 {
                                        continue;
                                    } else if (p as isize + r as isize - ph as isize) >= image_height as isize {
                                        continue;
                                    } else if (q as isize + s as isize - pw as isize) < 0 {
                                        continue;
                                    } else if (q as isize + s as isize - pw as isize) >= image_width as isize {
                                        continue;
                                    }

                                    let x_ = inp[
                                        input_channels * image_width * image_height * n
                                        + input_channels * image_width * (p+r-pw)
                                        + input_channels * (q+s-ph)
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

    fn with_activation<I, O, F>(
        activation: ActivationTypeDef,
        input_data_type: DataTypeDef,
        output_data_type: DataTypeDef,
        act: F
    )
    where I: Copy + Default + From<f32>,
          O: Copy + Default + From<f32>,
          f32: From<I>,
          f32: From<O>,
          F: Fn(f32) -> f32,
    {
        let kernel: Vec<_> = (0..(64*3*3*40))
            .map(|_i| 1.0 - 2.0 * thread_rng().gen::<f32>())
            .collect();
        let bias: Vec<_> = (0..64)
            .map(|_i| 5.0 - 10.0 * thread_rng().gen::<f32>())
            .collect();

        let layer_def = LayerDef {
            type_of: LayerTypeDef::Conv2D,
            input: vec! [
                VariableDef { id: 0, shape: vec! [2, 19, 19, 40], data_type: input_data_type }
            ],
            output: vec! [
                VariableDef { id: 1, shape: vec! [2, 19, 19, 64], data_type: output_data_type }
            ],
            arguments: Some(LayerArgumentsDef {
                kernel: Some(ConstantDef {
                    shape: vec! [64, 3, 3, 40],
                    value: ConstantValueDef {
                        inner: Arc::new(kernel.clone())
                    }
                }),
                bias: Some(ConstantDef {
                    shape: vec! [64],
                    value: ConstantValueDef {
                        inner: Arc::new(bias.clone())
                    }
                }),
                alpha: None,
                group_count: 1,
                activation
            })
        };
        let layer = Conv2D::new(&layer_def)
            .expect("Could not create conv2d layer");

        let (inputs, outputs) = run_layer::<I, O, _>(
            &layer_def,
            &layer
        );

        let input: Vec<f32> = inputs[0].iter().map(|&x| f32::from(x)).collect();
        let expected_output = conv2d(
            &input,
            &kernel,
            &bias,
            2, 19, 19, 64, 3, 3, 40
        );

        for (expected_outp, &outp) in expected_output.into_iter().zip(outputs[0].iter()) {
            let expected_outp = act(expected_outp);

            assert_approx_eq_prec(
                f32::from(outp),
                expected_outp,
                if output_data_type == DataTypeDef::Half {
                    0.02
                } else {
                    0.01
                }
            );
        }
    }

    #[test]
    fn conv2d_relu_half() {
        with_activation::<f16, f16, _>(
            ActivationTypeDef::ReLU,
            DataTypeDef::Half,
            DataTypeDef::Half,
            |x| if x > 0.0 { x } else { 0.0 }
        )
    }

    #[test]
    fn conv2d_relu_float() {
        with_activation::<f32, f32, _>(
            ActivationTypeDef::ReLU,
            DataTypeDef::Float,
            DataTypeDef::Float,
            |x| if x > 0.0 { x } else { 0.0 }
        )
    }

    #[test]
    fn conv2d_sigmoid_half() {
        with_activation::<f16, f16, _>(
            ActivationTypeDef::Sigmoid,
            DataTypeDef::Half,
            DataTypeDef::Half,
            |x| 1.0 / (1.0 + E.powf(-x))
        )
    }

    #[test]
    fn conv2d_sigmoid_float() {
        with_activation::<f32, f32, _>(
            ActivationTypeDef::Sigmoid,
            DataTypeDef::Float,
            DataTypeDef::Float,
            |x| 1.0 / (1.0 + E.powf(-x))
        )
    }

    #[test]
    fn conv2d_linear_half() {
        with_activation::<f16, f16, _>(
            ActivationTypeDef::Linear,
            DataTypeDef::Half,
            DataTypeDef::Half,
            |x| x
        )
    }

    #[test]
    fn conv2d_linear_float() {
        with_activation::<f32, f32, _>(
            ActivationTypeDef::Linear,
            DataTypeDef::Float,
            DataTypeDef::Float,
            |x| x
        )
    }
}