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
use std::ptr;

use nn::ffi::{cuda, cudnn};
use nn::ops::tensor::*;

/// Operator for performing a 2D convolution using cuDNN.
pub struct Concat {
    alpha: f32,
    beta: f32,

    pub input_1_cnhw: cudnn::TensorDescriptor,
    pub input_2_cnhw: cudnn::TensorDescriptor,
    pub output_cnhw: cudnn::TensorDescriptor,

    pub stream_1: cuda::Stream,
    pub stream_2: cuda::Stream,
    pub start: cuda::Event,
    pub finish_1: cuda::Event,
    pub finish_2: cuda::Event,
}

impl Concat {
    /// Set the correct type, shape, and scale for the operands of the
    /// operator `output = concat([input_1, input_2], axis=1)`.
    ///
    /// # Arguments
    ///
    /// * `input_1` -
    /// * `input_2` -
    /// * `output` -
    ///
    pub fn calibrate(
        input_1: &Tensor,
        input_2: &Tensor,
        output: &Tensor
    )
    {
        let shape_1 = input_1.get_shape();
        let shape_2 = input_2.get_shape();

        debug_assert!(shape_1[0] == shape_2[0]);
        debug_assert!(shape_1[2] == shape_2[2]);
        debug_assert!(shape_1[3] == shape_2[3]);

        debug_assert!(input_1.get_data_type() == input_2.get_data_type());
        debug_assert!(input_1.get_format() == input_2.get_format());

        output.set_shape(vec! [
            shape_1[0],
            shape_1[1] + shape_2[1],
            shape_1[2],
            shape_1[3]
        ]);
        output.set_data_type(
            input_1.get_data_type(),
            input_1.get_format()
        );
    }

    /// Returns a convolutional operator that is optimized for the given
    /// arguments.
    ///
    /// # Arguments
    ///
    /// * `handle` -
    /// * `alpha` -
    /// * `input` -
    /// * `weights` -
    /// * `beta` -
    /// * `blend` -
    /// * `bias` -
    /// * `output` -
    /// * `min_workspace_size` -
    ///
    pub fn new(
        input_1: &Tensor,
        input_2: &Tensor,
        output: &Tensor
    ) -> Concat
    {
        let mut c = Concat {
            alpha: 1.0,
            beta: 0.0,

            input_1_cnhw: ptr::null(),
            input_2_cnhw: ptr::null(),
            output_cnhw: ptr::null(),

            stream_1: ptr::null(),
            stream_2: ptr::null(),

            start: ptr::null(),
            finish_1: ptr::null(),
            finish_2: ptr::null(),
        };

        unsafe {
            let input_shape_1 = input_1.get_shape();
            let input_shape_2 = input_2.get_shape();
            let output_shape = output.get_shape();

            check!(cudnn::cudnnCreateTensorDescriptor(&mut c.input_1_cnhw));
            check!(cudnn::cudnnSetTensor4dDescriptorEx(
                c.input_1_cnhw,
                input_1.get_data_type(),
                input_shape_1[0],  // n
                input_shape_1[1],  // c
                input_shape_1[2],  // h
                input_shape_1[3],  // w
                input_shape_1[3] * input_shape_1[2],
                input_shape_1[3] * input_shape_1[2] * input_shape_1[0],
                input_shape_1[3],
                1,
            ));

            check!(cudnn::cudnnCreateTensorDescriptor(&mut c.input_2_cnhw));
            check!(cudnn::cudnnSetTensor4dDescriptorEx(
                c.input_2_cnhw,
                input_2.get_data_type(),
                input_shape_2[0],  // n
                input_shape_2[1],  // c
                input_shape_2[2],  // h
                input_shape_2[3],  // w
                input_shape_2[3] * input_shape_2[2],
                input_shape_2[3] * input_shape_2[2] * input_shape_2[0],
                input_shape_2[3],
                1,
            ));

            check!(cudnn::cudnnCreateTensorDescriptor(&mut c.output_cnhw));
            check!(cudnn::cudnnSetTensor4dDescriptorEx(
                c.output_cnhw,
                output.get_data_type(),
                output_shape[0],  // n
                output_shape[1],  // c
                output_shape[2],  // h
                output_shape[3],  // w
                output_shape[3] * output_shape[2],
                output_shape[3] * output_shape[2] * output_shape[0],
                output_shape[3],
                1,
            ));

            check!(cuda::cudaStreamCreate(&mut c.stream_1));
            check!(cuda::cudaStreamCreate(&mut c.stream_2));
            check!(cuda::cudaEventCreate(&mut c.start));
            check!(cuda::cudaEventCreate(&mut c.finish_1));
            check!(cuda::cudaEventCreate(&mut c.finish_2));
        }

        c
    }

    /// Performs the appropriate cuDNN calls to compute the concatenation of
    /// the two tensors.
    ///
    /// # Arguments
    ///
    /// * `handle` -
    /// * `input_1` - 
    /// * `output` -
    /// * `output_data` -
    ///
    pub fn forward(
        &self,
        handle: cudnn::Handle,
        input_1: &Tensor,
        input_data_1: *mut c_void,        
        input_2: &Tensor,
        input_data_2: *mut c_void,
        output: &Tensor,
        output_data: *mut c_void,        
        output_workspace: *mut c_void,        
    )
    {
        let input_workspace_1 = output_workspace;
        let input_workspace_2 = unsafe {
            output_workspace.offset(input_1.get_size_in_bytes() as isize)
        };

        let mut current_stream = ptr::null();

        unsafe {
            check!(cudnn::cudnnGetStream(handle, &mut current_stream));

            // transform `input_1` to CNHW
            check!(cudnn::cudnnSetStream(handle, self.stream_1));
            check!(cudnn::cudnnTransformTensor(
                handle,
                &self.alpha,
                input_1.tensor_desc, input_data_1,
                &self.beta,
                self.input_1_cnhw, input_workspace_1
            ));

            check!(cuda::cudaEventRecord(self.finish_1, self.stream_1));
            check!(cuda::cudaStreamWaitEvent(current_stream, self.finish_1, 0));

            // transform `input_2` to CNHW
            check!(cudnn::cudnnSetStream(handle, self.stream_2));
            check!(cudnn::cudnnTransformTensor(
                handle,
                &self.alpha,
                input_2.tensor_desc, input_data_2,
                &self.beta,
                self.input_2_cnhw, input_workspace_2
            ));

            check!(cuda::cudaEventRecord(self.finish_2, self.stream_2));
            check!(cuda::cudaStreamWaitEvent(current_stream, self.finish_2, 0));

            // transform `[input_1, input_2]` to NCHW
            check!(cudnn::cudnnSetStream(handle, current_stream));
            check!(cudnn::cudnnTransformTensor(
                handle,
                &self.alpha,
                self.output_cnhw, output_workspace,
                &self.beta,
                output.tensor_desc, output_data
            ));
        }
    }
}

impl Drop for Concat {
    fn drop(&mut self) {
        unsafe {
            cudnn::cudnnDestroyTensorDescriptor(self.input_1_cnhw);
            cudnn::cudnnDestroyTensorDescriptor(self.input_2_cnhw);
            cudnn::cudnnDestroyTensorDescriptor(self.output_cnhw);

            cuda::cudaStreamDestroy(self.stream_1);
            cuda::cudaStreamDestroy(self.stream_2);
            cuda::cudaEventDestroy(self.start);
            cuda::cudaEventDestroy(self.finish_1);
            cuda::cudaEventDestroy(self.finish_2);
        }
    }
}
