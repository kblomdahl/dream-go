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

use libc::c_void;

use nn::ffi::cudnn;
use nn::ops::Tensor;

/// Generic parameters for trivial layers such as RelU, Tanh, or Softmax
/// that does not warrent its own structure.
pub struct Operator {
    pub alpha: f32,
    pub beta: f32
}

/// Operator that implements the softmax operator
pub struct Softmax;

impl Softmax {
    /// Returns a new output tensor that has the correct type, shape, and scale
    /// for the output of the operator `softmax(input)`.
    /// 
    /// # Arguments
    /// 
    /// * `input` -
    /// 
    pub fn calibrate(input: &Tensor) -> Tensor {
        let output = Tensor::default();

        // softmax is not linear, so both the input and output must have a
        // scale of `1.0` of the results will be wrong.
        assert_eq!(input.get_scale(), 1.0);

        output.set_data_type_like(input);
        output.set_scale(1.0);
        output.set_shape_like(input);
        output
    }

    /// Returns a new softmax operator that does not blend the input with the
    /// output.
    pub fn new(input: &Tensor, output: &Tensor) -> Operator {
        Operator {
            alpha: input.get_scale() / output.get_scale(),
            beta: 0.0
        }
    }

    /// Perform the appropriate cuDNN calls to perform the softmax operation on
    /// the `input_data` and store the results in the `output_data`.
    /// 
    /// # Arguments
    /// 
    /// * `op` -
    /// * `handle` -
    /// * `input` -
    /// * `input_data` -
    /// * `output` -
    /// * `output_data` -
    /// 
    pub fn forward(
        op: &Operator,
        handle: cudnn::Handle,
        input: &Tensor,
        input_data: *const c_void,
        output: &Tensor,
        output_data: *mut c_void
    )
    {
        unsafe {
            check!(cudnn::cudnnSoftmaxForward(
                handle,
                cudnn::SoftmaxAlgorithm::Accurate,
                cudnn::SoftmaxMode::Instance,
                &op.alpha,  // alpha
                input.tensor_desc, input_data,  // input
                &op.beta,  // beta
                output.tensor_desc, output_data  // output
            ));
        }
    }
}

/// Operator that implements the RelU operator
pub struct Relu;

impl Relu {
    /// Returns a new output tensor that has the correct type, shape, and scale
    /// for the output of the operator `relu(input)`.
    /// 
    /// # Arguments
    /// 
    /// * `input` -
    /// 
    pub fn calibrate(input: &Tensor) -> Tensor {
        let output = Tensor::default();

        output.set_data_type_like(input);
        output.set_scale(1.0);
        output.set_shape_like(input);
        output
    }

    /// Returns a new RelU operator that does not blend the input with the
    /// output.
    pub fn new() -> Operator {
        Operator {
            alpha: 1.0,
            beta: 0.0
        }
    }

    /// Perform the appropriate cuDNN calls to perform the relu operation on
    /// the `input_data` and store the results in the `output_data`.
    /// 
    /// # Arguments
    /// 
    /// * `op` -
    /// * `handle` -
    /// * `relu` -
    /// * `input` -
    /// * `input_data` -
    /// * `output` -
    /// * `output_data` -
    /// 
    pub fn forward(
        op: &Operator,
        handle: cudnn::Handle,
        relu: cudnn::ActivationDescriptor,
        input: &Tensor,
        input_data: *const c_void,
        output: &Tensor,
        output_data: *mut c_void
    )
    {
        unsafe {
            check!(cudnn::cudnnActivationForward(
                handle,
                relu,
                &op.alpha,  // alpha
                input.tensor_desc, input_data,  // input
                &op.beta,  // beta
                output.tensor_desc, output_data,  // output
            ));
        }
    }
}

/// Operator that implements the Tanh operator
pub struct Tanh;

impl Tanh {
    /// Returns a new output tensor that has the correct type, shape, and scale
    /// for the output of the operator `tanh(input)`.
    /// 
    /// # Arguments
    /// 
    /// * `input` -
    /// 
    pub fn calibrate(input: &Tensor) -> Tensor {
        let output = Tensor::default();

        // tanh is not linear, so both the input and output must have a
        // scale of `1.0` of the results will be wrong.
        assert_eq!(input.get_scale(), 1.0);

        output.set_data_type_like(input);
        output.set_scale(1.0);
        output.set_shape_like(input);
        output
    }

    /// Returns a new Tanh operator that does not blend the input with the
    /// output.
    pub fn new() -> Operator {
        Operator {
            alpha: 1.0,
            beta: 0.0
        }
    }

    /// Perform the appropriate cuDNN calls to perform the tanh operation on
    /// the `input_data` and store the results in the `output_data`.
    /// 
    /// # Arguments
    /// 
    /// * `op` -
    /// * `handle` -
    /// * `tanh` -
    /// * `input` -
    /// * `input_data` -
    /// * `output` -
    /// * `output_data` -
    /// 
    pub fn forward(
        op: &Operator,
        handle: cudnn::Handle,
        tanh: cudnn::ActivationDescriptor,
        input: &Tensor,
        input_data: *const c_void,
        output: &Tensor,
        output_data: *mut c_void
    )
    {
        unsafe {
            check!(cudnn::cudnnActivationForward(
                handle,
                tanh,
                &op.alpha,  // alpha
                input.tensor_desc, input_data,  // input
                &op.beta,  // beta
                output.tensor_desc, output_data,  // output
            ));
        }
    }
}
