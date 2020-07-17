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

use std::collections::HashMap;
use std::mem::size_of;
use std::ptr;
use std::rc::Rc;
use std::sync::Arc;

use dg_cuda as cuda2;
use dg_cuda::cudnn as cudnn2;
use dg_go::utils::features::{FEATURE_SIZE, NUM_FEATURES};
use dg_utils::types::f16;
use dg_utils::config;
use super::ffi::cuda;
use super::output_map::*;
use super::tensor::Tensor;
use super::Error;

/// The number of channels to assume if not given in the network weights file.
const DEFAULT_NUM_CHANNELS: i32 = 128;

// -------- InferenceType --------

pub trait InferenceType: Copy + Default + Sized {
    type Tower: InferenceType;
    type Output: InferenceType;

    fn as_f32(self) -> f32;
}

impl InferenceType for f16 {
    type Tower = f16;
    type Output = f16;

    fn as_f32(self) -> f32 { f32::from(self) }
}

impl InferenceType for f32 {
    type Tower = f32;
    type Output = f32;

    fn as_f32(self) -> f32 { self }
}

/// If the given output `output` is in the set of requested outputs, then
/// loads the given pointers from the device to the host, and then adds it
/// to the output map.
///
/// # Arguments
///
/// * `output_set` - The set of requested outputs.
/// * `output_map` - The outputs.
/// * `output` - The output to check for.
/// * `device_ptr` -
/// * `num_elements`-
/// * `stream` -
///
unsafe fn load_output<A: cuda2::Allocator, T: InferenceType>(
    output_set: &OutputSet,
    output_map: &mut OutputMap<Vec<f32>>,
    output: Output,
    ptr: &cuda2::SmartPtr<A>,
    stream: &cuda2::Stream
) -> Result<(), Error>
{
    if let Some(key) = output_set.contains(output) {
        output_map.put(
            key,
            ptr.to_vec::<T::Output>(&stream)?.iter()
                .map(|x| x.as_f32())
                .collect()
        );
    }

    Ok(())
}

/// Returns true if the current device supports `f16` (in a
/// sensible way).
fn has_true_half() -> bool {
    let mut version_major: i32 = 0;
    let mut version_minor: i32 = 0;

    unsafe {
        assert!(cuda::cudaDeviceGetAttribute(&mut version_major, cuda::DeviceAttr::ComputeCapabilityMajor, 0).is_ok());
        assert!(cuda::cudaDeviceGetAttribute(&mut version_minor, cuda::DeviceAttr::ComputeCapabilityMinor, 0).is_ok());
    }

    (version_major == 6 && version_minor == 0) ||
        (version_major == 6 && version_minor == 2) ||
        (version_major >= 7)
}

/// Returns a `TensorDescriptor` for an feature tensor for the given
/// `batch_size` and `num_channels`.
/// 
/// # Arguments
/// 
/// * `batch_size` -
/// * `num_channels` -
/// 
fn create_tensor_descriptor(batch_size: i32, num_channels: i32) -> Result<cudnn2::TensorDescriptor, cudnn2::Status> {
    cudnn2::TensorDescriptor::new(
        cudnn2::TensorFormat::NHWC,
        cudnn2::DataType::Half,
        &[batch_size, num_channels, 19, 19]
    )
}

/// Returns a `TensorDescriptor` for an offset tensor for the given
/// `num_channels`.
/// 
/// # Arguments
/// 
/// * `num_channels` -
/// 
fn create_offset_descriptor(num_channels: i32) -> Result<cudnn2::TensorDescriptor, cudnn2::Status> {
    cudnn2::TensorDescriptor::new(
        cudnn2::TensorFormat::NHWC,
        cudnn2::DataType::Half,
        &[1, num_channels, 1, 1]
    )
}

/// Returns a `TensorDescriptor` for a dense tensor for the given `batch_size`
/// and `size`.
///
/// # Arguments
/// 
/// * `batch_size` -
/// * `size` -
/// 
fn create_dense_descriptor(batch_size: i32, size: i32) -> Result<cudnn2::TensorDescriptor, cudnn2::Status> {
    cudnn2::TensorDescriptor::new(
        cudnn2::TensorFormat::NHWC,
        cudnn2::DataType::Half,
        &[batch_size, size, 1, 1]
    )
}

/// Returns a `FilterDescriptor` for a three wide and high filter for the given
/// `num_outputs` and `num_inputs` features.
/// 
/// # Arguments
/// 
/// * `num_outputs` -
/// * `num_inputs` -
/// 
fn create_filter_descriptor_3x3(num_outputs: i32, num_inputs: i32) -> Result<cudnn2::FilterDescriptor, cudnn2::Status> {
    cudnn2::FilterDescriptor::new(
        cudnn2::DataType::Half,
        cudnn2::TensorFormat::NHWC,
        &[num_outputs, num_inputs, 3, 3]
    )
}

/// Returns a `FilterDescriptor` for a one wide and high filter for the given
/// `num_outputs` and `num_inputs` features.
/// 
/// # Arguments
/// 
/// * `num_outputs` -
/// * `num_inputs` -
/// 
fn create_filter_descriptor_1x1(num_outputs: i32, num_inputs: i32) -> Result<cudnn2::FilterDescriptor, cudnn2::Status> {
    cudnn2::FilterDescriptor::new(
        cudnn2::DataType::Half,
        cudnn2::TensorFormat::NCHW,
        &[num_outputs, num_inputs, 1, 1]
    )
}

/// Returns a `ConvolutionDescriptor` for a three wide and high filter.
fn create_convolution_descriptor_3x3() -> Result<cudnn2::ConvolutionDescriptor, cudnn2::Status> {
    cudnn2::ConvolutionDescriptor::new(
        &[1, 1],
        &[1, 1],
        &[1, 1],
        cudnn2::ConvolutionMode::CrossCorrelation,
        if has_true_half() { cudnn2::DataType::Half } else { cudnn2::DataType::Float }
    )
}

/// Returns a `ConvolutionDescriptor` for a one wide and high filter using a
/// 32-bit compute type.
fn create_convolution_descriptor_1x1_float() -> Result<cudnn2::ConvolutionDescriptor, cudnn2::Status> {
    cudnn2::ConvolutionDescriptor::new(
        &[0, 0],
        &[1, 1],
        &[1, 1],
        cudnn2::ConvolutionMode::CrossCorrelation,
        cudnn2::DataType::Float
    )
}

/// Returns a `ConvolutionBiasActivation` for a three wide and high
/// convolution, add bias, and activation operation for the given
/// `batch_size`, `num_outputs`, and `num_inputs`.
/// 
/// # Arguments
/// 
/// * `handle` -
/// * `batch_size` -
/// * `num_outputs` -
/// * `num_inputs` - 
/// 
fn create_convolution_bias_3x3(
    handle: &cudnn2::Handle,
    batch_size: i32,
    num_outputs: i32,
    num_inputs: i32,
    alpha: &[f32]
) -> Result<cudnn2::ConvolutionBiasActivation, cudnn2::Status>
{
    debug_assert_eq!(alpha.len(), 2);

    cudnn2::ConvolutionBiasActivation::new(
        handle,
        alpha[0],
        create_tensor_descriptor(batch_size, num_inputs)?,
        create_filter_descriptor_3x3(num_outputs, num_inputs)?,
        create_convolution_descriptor_3x3()?,
        alpha[1],
        create_offset_descriptor(num_outputs)?,
        cudnn2::ActivationDescriptor::identity()?,
        create_tensor_descriptor(batch_size, num_outputs)?
    )
}

/// Returns a `ConvolutionBiasActivation` for a three wide and high
/// convolution, add bias, and activation operation for the given
/// `batch_size`, `num_outputs`, and `num_inputs`.
/// 
/// # Arguments
/// 
/// * `handle` -
/// * `batch_size` -
/// * `num_outputs` -
/// * `num_inputs` - 
/// 
fn create_convolution_bias_activation_3x3(
    handle: &cudnn2::Handle,
    batch_size: i32,
    num_outputs: i32,
    num_inputs: i32,
    alpha: &[f32]
) -> Result<cudnn2::ConvolutionBiasActivation, cudnn2::Status>
{
    debug_assert_eq!(alpha.len(), 2);

    cudnn2::ConvolutionBiasActivation::new(
        handle,
        alpha[0],
        create_tensor_descriptor(batch_size, num_inputs)?,
        create_filter_descriptor_3x3(num_outputs, num_inputs)?,
        create_convolution_descriptor_3x3()?,
        alpha[1],
        create_offset_descriptor(num_outputs)?,
        cudnn2::ActivationDescriptor::relu()?,
        create_tensor_descriptor(batch_size, num_outputs)?
    )
}

/// Returns a `ConvolutionBiasActivation` for a one wide and high
/// convolution, add bias, and activation operation for the given
/// `batch_size`, `num_outputs`, and `num_inputs`.
/// 
/// # Arguments
/// 
/// * `handle` -
/// * `batch_size` -
/// * `num_outputs` -
/// * `num_inputs` - 
/// 
fn create_dense_bias_activation(
    handle: &cudnn2::Handle,
    batch_size: i32,
    num_outputs: i32,
    num_inputs: i32,
    alpha: &[f32],
    activation: cudnn2::ActivationDescriptor
) -> Result<cudnn2::ConvolutionBiasActivation, cudnn2::Status>
{
    debug_assert_eq!(alpha.len(), 2);

    cudnn2::ConvolutionBiasActivation::new(
        handle,
        alpha[0],
        create_dense_descriptor(batch_size, num_inputs)?,
        create_filter_descriptor_1x1(num_outputs, num_inputs)?,
        create_convolution_descriptor_1x1_float()?,
        alpha[1],
        create_offset_descriptor(num_outputs)?,
        activation,
        create_dense_descriptor(batch_size, num_outputs)?
    )
}

/// Returns a `ConvolutionBiasActivation` for a one wide and high
/// convolution, add bias, and activation operation for the given
/// `batch_size`, `num_outputs`, and `num_inputs`.
/// 
/// # Arguments
/// 
/// * `handle` -
/// * `batch_size` -
/// * `num_outputs` -
/// * `num_inputs` - 
/// 
fn create_dense_bias(
    handle: &cudnn2::Handle,
    batch_size: i32,
    num_outputs: i32,
    num_inputs: i32,
    alpha: &[f32]
) -> Result<cudnn2::ConvolutionBiasActivation, cudnn2::Status>
{
    create_dense_bias_activation(
        handle,
        batch_size,
        num_outputs,
        num_inputs,
        alpha,
        cudnn2::ActivationDescriptor::identity()?
    )
}

/// Returns a `Softmax` structure for the given `batch_size` and `num_channels`.
/// 
/// # Arguments
/// 
/// * `batch_size` -
/// * `num_channels` -
/// 
fn create_softmax(batch_size: i32, num_channels: i32) -> Result<cudnn2::Softmax, cudnn2::Status> {
    cudnn2::Softmax::new(
        cudnn2::SoftmaxMode::Instance,
        create_dense_descriptor(batch_size, num_channels)?,
        create_dense_descriptor(batch_size, num_channels)?,
        &[1.0, 0.0]
    )
}

/// Returns a `Transform` structure that transposes a 2D tensor with the given
/// `num_outputs` and `num_inputs` from the shape `[num_inputs, num_outputs]` to
/// `[num_outputs, num_inputs]`.
/// 
/// # Arguments
/// 
/// * `num_outputs` -
/// * `num_inputs` -
/// 
fn create_dense_transpose(num_outputs: i32, num_inputs: i32) -> Result<cudnn2::Transform, cudnn2::Status> {
    cudnn2::Transform::new(
        cudnn2::TensorDescriptor::new_ex(
            cudnn2::DataType::Half,
            &[num_outputs, num_inputs, 1, 1],
            &[1, num_outputs, 1, 1],
        )?,
        cudnn2::TensorDescriptor::new_ex(
            cudnn2::DataType::Half,
            &[num_outputs, num_inputs, 1, 1],
            &[num_inputs, 1, 1, 1],
        )?,
        &[1.0, 0.0]
    )
}

/// Returns a `ReduceTensor` structure that reduce a `[n, 19, 19, c]` tensor
/// into a `[n, 1, 1, c]` tensor using average.
/// 
/// # Arguments
/// 
/// * `n` - 
/// * `num_channels` -
/// * `alpha` - 
/// 
fn create_global_avg_pooling(n: i32, num_channels: i32, alpha: [f32; 2]) -> Result<cudnn2::ReduceTensor, cudnn2::Status> {
    cudnn2::ReduceTensor::new(
        cudnn2::ReduceTensorDescriptor::new(
            cudnn2::ReduceTensorOp::Avg,
            cudnn2::DataType::Float,
            cudnn2::NanPropagation::NotPropagateNaN,
            cudnn2::ReduceTensorIndices::NoIndices,
            cudnn2::IndicesType::_32
        )?,
        create_tensor_descriptor(n, num_channels)?,
        create_dense_descriptor(n, 1)?,
        alpha
    )
}

/// Returns an `Activation` structure that performs an element-wise tanh
/// activation on the entire tensor.
/// 
/// # Arguments
/// 
/// * `activation_desc` - 
/// * `n` - 
/// * `num_acts` - 
/// * `alpha` - 
/// 
fn create_dense_tanh(
    activation_desc: cudnn2::ActivationDescriptor,
    n: i32,
    num_acts: i32,
    alpha: [f32; 2]
) -> Result<cudnn2::Activation, cudnn2::Status> {
    cudnn2::Activation::new(
        activation_desc,
        create_dense_descriptor(n, num_acts)?,
        create_dense_descriptor(n, num_acts)?,
        alpha
    )
}

// -------- Convolution --------

pub struct Conv2d {
    conv_desc: cudnn2::ConvolutionBiasActivation,
    filter: Tensor,
    offset: Tensor
}

impl Conv2d {
    fn new(
        name: String,
        conv_desc: cudnn2::ConvolutionBiasActivation,
        tensors: &HashMap<String, Tensor>
    ) -> Result<Self, Error>
    {
        Ok(Self {
            conv_desc: conv_desc,
            filter: tensors[&format!("{}:0", name)].clone(),
            offset: tensors[&format!("{}/offset:0", name)].clone()
        })
    }

    unsafe fn prepare(&self, handle: &cudnn2::Handle, stream: &cuda2::Stream) -> Result<bool, Error> {
        handle.set_stream(stream)?;
        if self.filter.copy_to_device(&stream)? && self.offset.copy_to_device(&stream)? {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    unsafe fn forward<A: cuda2::Allocator + Clone>(
        &self,
        handle: &cudnn2::Handle,
        input: &cuda2::SmartPtr<A>,
        allocator: &A,
        stream: &cuda2::Stream
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        let workspace = cuda2::malloc(self.conv_desc.fwd_algo_perf().memory(), allocator)?;
        let output = cuda2::malloc(self.conv_desc.output().size_in_bytes()?, allocator)?;

        self.prepare(handle, stream)?;
        self.conv_desc.forward(
            handle,
            input.as_ptr(),
            self.filter.get().as_ptr(),
            workspace.as_ptr(), workspace.size_in_bytes(),
            output.as_ptr(),
            self.offset.get().as_ptr(),
            output.as_ptr()
        )?;

        Ok(output)
    }

    unsafe fn forward_skip<A: cuda2::Allocator + Clone>(
        &self,
        handle: &cudnn2::Handle,
        input: &cuda2::SmartPtr<A>,
        skip_input: &cuda2::SmartPtr<A>,
        allocator: &A,
        stream: &cuda2::Stream
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        let workspace = cuda2::malloc(self.conv_desc.fwd_algo_perf().memory(), allocator)?;
        let output = cuda2::malloc(self.conv_desc.output().size_in_bytes()?, allocator)?;

        self.prepare(handle, stream)?;
        self.conv_desc.forward(
            handle,
            input.as_ptr(),
            self.filter.get().as_ptr(),
            workspace.as_ptr(), workspace.size_in_bytes(),
            skip_input.as_ptr(),
            self.offset.get().as_ptr(),
            output.as_ptr()
        )?;

        Ok(output)
    }
}

pub struct Dense {
    transpose_filter: cudnn2::Transform,
    conv_desc: cudnn2::ConvolutionBiasActivation,
    filter: Tensor,
    offset: Tensor,
}

impl Dense {
    fn new(
        name: String,
        conv_desc: cudnn2::ConvolutionBiasActivation,
        tensors: &HashMap<String, Tensor>
    ) -> Result<Self, Error>
    {
        let num_outputs = conv_desc.output().shape()?[1];
        let num_inputs = conv_desc.input().shape()?[1];

        Ok(Self {
            transpose_filter: create_dense_transpose(num_outputs, num_inputs)?,
            conv_desc: conv_desc,
            filter: tensors[&format!("{}:0", name)].clone(),
            offset: tensors[&format!("{}/offset:0", name)].clone(),
        })
    }

    unsafe fn prepare<A: cuda2::Allocator + Clone>(
        &self,
        handle: &cudnn2::Handle,
        allocator: &A,
        stream: &cuda2::Stream
    ) -> Result<bool, Error> 
    {
        handle.set_stream(stream)?;
        if self.filter.copy_to_device(&stream)? && self.offset.copy_to_device(&stream)? {
            let temp = cuda2::malloc(self.filter.get().size_in_bytes(), allocator)?;

            self.transpose_filter.forward(
                handle,
                self.filter.get().as_ptr(),
                temp.as_ptr()
            )?;
            self.filter.set_device_ptr(temp.unwrap());

            Ok(true)
        } else {
            Ok(false)
        }
    }

    unsafe fn forward<A: cuda2::Allocator + Clone>(
        &self,
        handle: &cudnn2::Handle,
        input: &cuda2::SmartPtr<A>,
        allocator: &A,
        stream: &cuda2::Stream
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        let workspace = cuda2::malloc(self.conv_desc.fwd_algo_perf().memory(), allocator)?;
        let output = cuda2::malloc(self.conv_desc.output().size_in_bytes()?, allocator)?;

        self.prepare(handle, allocator, stream)?;
        self.conv_desc.forward(
            handle,
            input.as_ptr(),
            self.filter.get().as_ptr(),
            workspace.as_ptr(), workspace.size_in_bytes(),
            output.as_ptr(),
            self.offset.get().as_ptr(),
            output.as_ptr()
        )?;

        Ok(output)
    }
}

// -------- Graph --------

pub struct Builder {
    tensors: Arc<HashMap<String, Tensor>>,
    allocator: cuda2::PerDevice<cuda2::Concurrent<cuda2::Sticky<cuda2::Native>>>,
}

impl Builder {
    pub fn new(tensors: HashMap<String, Tensor>) -> Builder {
        Builder {
            tensors: Arc::new(tensors),
            allocator: cuda2::PerDevice::new().unwrap(),
        }
    }

    /// Returns a mutable workspace that contains everything you need to
    /// perform a forward pass through the network pre-allocated.
    ///
    /// # Arguments
    ///
    /// * `batch_size` -
    ///
    pub fn get_workspace(&self, batch_size: usize) -> Result<Workspace, Error> {
        let handle_dnn: cudnn2::Handle = cudnn2::Handle::new()?;
        let c_up = unsafe { Rc::new(UpLayer::new(&handle_dnn, batch_size as i32, &self.tensors)?) };
        let c_residual = unsafe { self.get_residual_layers(&handle_dnn, batch_size)? };
        let c_value = unsafe { Rc::new(ValueLayer::new(&handle_dnn, batch_size as i32, 2 + c_residual.len(), &self.tensors)?) };
        let c_policy = unsafe { Rc::new(PolicyLayer::new(&handle_dnn, batch_size as i32, 2 + c_residual.len(), &self.tensors)?) };

        Ok(Workspace {
            batch_size: batch_size,
            allocator: self.allocator.clone(),

            handle: handle_dnn,

            tower_finished: cuda2::Event::new()?,

            tower_stream: cuda2::Stream::new()?,
            policy_stream: cuda2::Stream::new()?,
            value_stream: cuda2::Stream::new()?,

            c_up: c_up,
            c_value: c_value,
            c_policy: c_policy,
            c_residual: c_residual
        })
    }

    unsafe fn get_residual_layers(
        &self,
        handle_dnn: &cudnn2::Handle,
        batch_size: usize
    ) -> Result<Vec<Rc<ResidualLayer>>, Error>
    {
        let mut c_residual = vec! [];
        let mut count = 2;

        loop {
            match ResidualLayer::new(handle_dnn, batch_size as i32, count, &self.tensors) {
                Ok(None) => { break },
                Ok(Some(layer)) => { c_residual.push(Rc::new(layer)) },
                Err(reason) => { return Err(reason) }
            }

            count += 1;
        }

        Ok(c_residual)
    }
}

pub struct Workspace {
    batch_size: usize,
    allocator: cuda2::Concurrent<cuda2::Sticky<cuda2::Native>>,

    handle: cudnn2::Handle,
    tower_finished: cuda2::Event,
    tower_stream: cuda2::Stream,
    policy_stream: cuda2::Stream,
    value_stream: cuda2::Stream,

    c_up: Rc<UpLayer>,
    c_value: Rc<ValueLayer>,
    c_policy: Rc<PolicyLayer>,
    c_residual: Vec<Rc<ResidualLayer>>
}

struct UpLayer {
    up: Conv2d
}

impl UpLayer {
    /// Create a single convolutional layer.
    ///
    /// # Arguments
    ///
    /// * `handle` - The cuDNN handle
    /// * `n` - The number of images.
    /// * `tensors` -
    ///
    unsafe fn new(handle: &cudnn2::Handle, n: i32, tensors: &HashMap<String, Tensor>) -> Result<UpLayer, Error> {
        let num_channels = tensors.get("num_channels:0")
            .map(|x| { x.as_i32() })
            .unwrap_or(DEFAULT_NUM_CHANNELS);

        Ok(UpLayer {
            up: Conv2d::new("01_upsample/conv_1".into(), create_convolution_bias_activation_3x3(handle, n, num_channels, NUM_FEATURES as i32, &[1.0, 0.0])?, tensors)?
        })
    }

    unsafe fn forward<'a, A: cuda2::Allocator + Clone, T: InferenceType>(
        &self,
        workspace: &mut Workspace,
        allocator: &mut A,
        input: &cuda2::SmartPtr<A>
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        self.up.forward(&workspace.handle, input, allocator, &workspace.tower_stream)
    }
}

struct ResidualLayer {
    conv_1: Conv2d,
    conv_2: Conv2d,
    scale_offset: cudnn2::Scale
}

impl ResidualLayer {
    /// Create a layer that takes the final output of the residual block and
    /// transforms it into a scalar value.
    ///
    /// # Arguments
    ///
    /// * `handle` - The cuDNN handle
    /// * `n` - The number of images.
    /// * `i` - The index of the layer.
    /// * `tensors` -
    ///
    unsafe fn new(handle: &cudnn2::Handle, n: i32, i: usize, tensors: &HashMap<String, Tensor>) -> Result<Option<ResidualLayer>, Error> {
        let weights_1 = tensors.get(&format!("{:02}_residual/conv_1:0", i));
        let weights_2 = tensors.get(&format!("{:02}_residual/conv_2:0", i));
        let alpha = tensors.get(&format!("{:02}_residual/alpha:0", i));

        if weights_1.is_none() || weights_2.is_none() {
            return Ok(None);
        }

        let num_channels = tensors.get("num_channels:0")
            .map(|x| { x.as_i32() })
            .unwrap_or(DEFAULT_NUM_CHANNELS);
        let gate_t = alpha.map(|t| t.as_f32()).unwrap_or(0.5);

        Ok(Some(ResidualLayer {
            conv_1: Conv2d::new(format!("{:02}_residual/conv_1", i), create_convolution_bias_activation_3x3(handle, n, num_channels, num_channels, &[1.0, 0.0])?, tensors)?,
            conv_2: Conv2d::new(format!("{:02}_residual/conv_2", i), create_convolution_bias_activation_3x3(handle, n, num_channels, num_channels, &[gate_t, 1.0 - gate_t])?, tensors)?,
            scale_offset: cudnn2::Scale::new(create_offset_descriptor(num_channels)?, gate_t)?
        }))
    }

    unsafe fn forward<'a, A: cuda2::Allocator + Clone, T: InferenceType>(
        &self,
        workspace: &mut Workspace,
        allocator: &mut A,
        input: cuda2::SmartPtr<A>
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        if self.conv_2.prepare(&workspace.handle, &workspace.tower_stream)? {
            self.scale_offset.forward(&workspace.handle, self.conv_2.offset.get().as_ptr())?;
        }

        let y = self.conv_1.forward(&workspace.handle, &input, allocator, &workspace.tower_stream)?;
        self.conv_2.forward_skip(&workspace.handle, &y, &input, allocator, &workspace.tower_stream)
    }
}

struct ValueLayer {
    conv_1: Conv2d,
    reduce_mean: cudnn2::ReduceTensor,
    tanh: cudnn2::Activation,
}

impl ValueLayer {
    /// Create a layer that takes the final output of the residual block and
    /// transforms it into a scalar value.
    ///
    /// # Arguments
    ///
    /// * `handle` - The cuDNN handle
    /// * `n` - The number of images.
    /// * `i` - The index of the layer.
    /// * `tensors` -
    ///
    unsafe fn new(handle: &cudnn2::Handle, n: i32, i: usize, tensors: &HashMap<String, Tensor>) -> Result<ValueLayer, Error> {
        let num_channels = tensors.get("num_channels:0")
            .map(|x| { x.as_i32() })
            .unwrap_or(DEFAULT_NUM_CHANNELS);

        Ok(ValueLayer {
            conv_1: Conv2d::new(format!("{:02}v_value/conv_1", i), create_convolution_bias_3x3(handle, n, 8, num_channels, &[1.0, 0.0])?, tensors)?,
            reduce_mean: create_global_avg_pooling(n, 8, [1.0, 0.0])?,
            tanh: create_dense_tanh(cudnn2::ActivationDescriptor::tanh()?, n, 1, [1.0, 0.0])?
        })
    }

    unsafe fn forward<'a, A: cuda2::Allocator + Clone, T: InferenceType>(
        &self,
        workspace: &mut Workspace,
        allocator: &mut A,
        output_set: &OutputSet,
        output_map: &mut OutputMap<Vec<f32>>,
        input: &cuda2::SmartPtr<A>
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        workspace.handle.set_stream(&workspace.value_stream)?;

        // perform the forward convolution
        let workspace_v_size = self.reduce_mean.size_in_bytes(&workspace.handle)?;
        let workspace_v = cuda2::malloc(workspace_v_size, allocator)?;
        let value_1 = self.conv_1.forward(&workspace.handle, input, allocator, &workspace.value_stream)?;

        load_output::<_, T::Output>(output_set, output_map, Output::ValueDown, &value_1, &workspace.value_stream)?;

        // perform the global average pooling
        let value_2 = cuda2::malloc(self.reduce_mean.output().size_in_bytes()?, allocator)?;

        self.reduce_mean.forward(
            &workspace.handle,
            ptr::null_mut(), 0,
            workspace_v.as_ptr(), workspace_v.size_in_bytes(),
            value_1.as_ptr(),
            value_2.as_ptr()
        )?;

        load_output::<_, T::Output>(output_set, output_map, Output::ValueGemm, &value_2, &workspace.value_stream)?;

        // perform the feed-forward linear layer (tanh)
        self.tanh.forward(&workspace.handle, value_2.as_ptr(), value_2.as_ptr())?;

        Ok(value_2)
    }
}

struct PolicyLayer {
    conv_1: Conv2d,
    linear_2: Dense,
    softmax: cudnn2::Softmax,
    scale_tau: cudnn2::Scale,
}

impl PolicyLayer {
    /// Create a layer that takes the final output of the residual block and
    /// transforms it into a policy vector.
    ///
    /// # Arguments
    ///
    /// * `handle` - The cuDNN handle
    /// * `n` - The number of images.
    /// * `i` - The index of the layer.
    /// * `tensors` -
    ///
    unsafe fn new(handle: &cudnn2::Handle, n: i32, i: usize, tensors: &HashMap<String, Tensor>) -> Result<PolicyLayer, Error> {
        let num_channels = tensors.get("num_channels:0")
            .map(|x| { x.as_i32() })
            .unwrap_or(DEFAULT_NUM_CHANNELS);
        let num_samples = 8;
        let tau = 1.0 / *config::SOFTMAX_TEMPERATURE;

        Ok(PolicyLayer {
            conv_1: Conv2d::new(format!("{:02}p_policy/conv_1", i), create_convolution_bias_activation_3x3(handle, n, num_samples, num_channels, &[1.0, 0.0])?, tensors)?,
            linear_2: Dense::new(format!("{:02}p_policy/linear_1", i), create_dense_bias(handle, n, 362, 361 * num_samples, &[tau, 0.0])?, tensors)?,
            softmax: create_softmax(n, 362)?,
            scale_tau: cudnn2::Scale::new(create_offset_descriptor(362)?, tau)?,
        })
    }

    unsafe fn forward<'a, A: cuda2::Allocator + Clone, T: InferenceType>(
        &self,
        workspace: &mut Workspace,
        allocator: &mut A,
        output_set: &OutputSet,
        output_map: &mut OutputMap<Vec<f32>>,
        input: &cuda2::SmartPtr<A>
    ) -> Result<cuda2::SmartPtr<A>, Error>
    {
        if self.linear_2.prepare(&workspace.handle, allocator, &workspace.policy_stream)? {
            self.scale_tau.forward(&workspace.handle, self.linear_2.offset.get().as_ptr())?;
        }

        // perform the forward convolution
        let policy_1 = self.conv_1.forward(&workspace.handle, input, allocator, &workspace.policy_stream)?;

        load_output::<_, T::Output>(output_set, output_map, Output::PolicyDown, &policy_1, &workspace.policy_stream)?;

        // perform the feed-forward linear layers
        let policy_2 = self.linear_2.forward(&workspace.handle, &policy_1, allocator, &workspace.policy_stream)?;
        let policy_3 = cuda2::malloc(policy_2.size_in_bytes(), allocator)?;

        // softmax activation
        self.softmax.forward(&workspace.handle, policy_2.as_ptr(), policy_3.as_ptr())?;

        Ok(policy_3)
    }
}

/// Returns the value and policy tensors obtained from a forward pass
/// through the neural network.
///
/// # Arguments
///
/// * `workspace` - the workspace for the current thread
/// * `features` - the input features
/// * `outputs` - the outputs to copy to host memory
///
pub fn forward<T: InferenceType>(
    workspace: &mut Workspace,
    features: &[T],
    outputs: OutputSet
) -> Result<OutputMap<Vec<f32>>, Error>
{
    debug_assert!(features.len() % FEATURE_SIZE == 0);
    debug_assert!(features.len() / FEATURE_SIZE == workspace.batch_size);

    let mut allocator = cuda2::Cloneable::new(cuda2::Sticky::new(workspace.allocator.clone()));
    let mut map = OutputMap::default();

    unsafe {
        workspace.handle.set_stream(&workspace.tower_stream)?;

        // copy all of the input features into a temporary workspace
        let mut input = cuda2::malloc(size_of::<T>() * features.len(), &allocator)?;
        input.copy_from_slice(&features, &workspace.tower_stream)?;

        // Upsample 32 -> 128 channels
        let mut residual_1 = workspace.c_up.clone().forward::<_, T>(workspace, &mut allocator, &input)?;

        load_output::<_, T::Tower>(&outputs, &mut map, Output::Upsample, &residual_1, &workspace.tower_stream)?;

        // residual blocks
        let num_residual = workspace.c_residual.len();

        for i in 0..num_residual {
            let residual = &workspace.c_residual[i];
            let output = ::std::mem::transmute(Output::Residual_00 as u8 + i as u8);

            residual_1 = residual.clone().forward::<_, T>(workspace, &mut allocator, residual_1)?;
            load_output::<_, T::Tower>(&outputs, &mut map, output, &residual_1, &workspace.tower_stream)?;
        }

        workspace.tower_finished.record(&workspace.tower_stream)?;
        workspace.value_stream.wait_event(&workspace.tower_finished)?;
        workspace.policy_stream.wait_event(&workspace.tower_finished)?;

        // run the value and policy head, then wait for them to finish (if
        // they are requested)
        let value = workspace.c_value.clone().forward::<_, T>(workspace, &mut allocator, &outputs, &mut map, &residual_1)?;
        let policy = workspace.c_policy.clone().forward::<_, T>(workspace, &mut allocator, &outputs, &mut map, &residual_1)?;

        load_output::<_, T::Output>(&outputs, &mut map, Output::Value, &value, &workspace.value_stream)?;
        load_output::<_, T::Output>(&outputs, &mut map, Output::Policy, &policy, &workspace.policy_stream)?;
    }

    // pretty-print the tensor to stderr if logging is turned on
    if cfg!(feature = "trace-cuda") {
        for name in outputs.iter() {
            let mut dbg = String::new();
            let host = map.get(name);

            for (i, value) in host.iter().enumerate() {
                if i > 0 { dbg += ", "; }

                dbg += &format!("{:.4}", value);
            }

            eprintln!("{:?} = [{}]", name, dbg);
        }
    }

    Ok(map)
}
