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

use libc::c_void;

use dg_cuda as cuda2;
use dg_cuda::cudnn as cudnn2;
use dg_go::utils::features::{FEATURE_SIZE, NUM_FEATURES};
use dg_utils::types::f16;
use dg_utils::config;
use super::devices::get_current_device;
use super::ffi::{cublas, cuda, cudnn};
use super::slots::*;
use super::output_map::*;
use super::tensor::Tensor;
use super::Error;

/// A __global__ constant that contains `0.0`.
const ZERO: f32 = 0.0;

/// A __global__ constant that contains `1.0`.
const ONE: f32 = 1.0;

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

/// Copy the value of the given tensor from the device to the host.
///
/// # Arguments
///
/// * `ptr` - the memory address on the device
/// * `num_elements` - the number of elements to copy
/// * `stream` - the stream to execute the copy on
///
unsafe fn load_to_host<T: InferenceType>(
    ptr: *const c_void,
    num_elements: usize,
    stream: cuda::Stream
) -> Result<Vec<f32>, Error>
{
    let mut host = vec! [T::default(); num_elements];

    check!(cuda::cudaMemcpyAsync(
        host.as_mut_ptr() as *mut c_void,
        ptr,
        size_of::<T>() * num_elements,
        cuda::MemcpyKind::DeviceToHost,
        stream
    ))?;
    check!(cuda::cudaStreamSynchronize(stream))?;

    Ok(host.into_iter().map(|x| x.as_f32()).collect())
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
unsafe fn load_output<T: InferenceType>(
    output_set: &OutputSet,
    output_map: &mut OutputMap<Vec<f32>>,
    output: Output,
    device_ptr: *const c_void,
    num_elements: usize,
    stream: cuda::Stream
) -> Result<(), Error>
{
    if let Some(key) = output_set.contains(output) {
        output_map.put(key, load_to_host::<T>(
            device_ptr,
            num_elements,
            stream
        )?);
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
        cudnn2::TensorFormat::NHWC,
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

/// Returns a `ConvolutionDescriptor` for a one wide and high filter.
fn create_convolution_descriptor_1x1() -> Result<cudnn2::ConvolutionDescriptor, cudnn2::Status> {
    cudnn2::ConvolutionDescriptor::new(
        &[0, 0],
        &[1, 1],
        &[1, 1],
        cudnn2::ConvolutionMode::CrossCorrelation,
        if has_true_half() { cudnn2::DataType::Half } else { cudnn2::DataType::Float }
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
fn create_convolution_bias_activation_1x1(
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
        create_filter_descriptor_1x1(num_outputs, num_inputs)?,
        create_convolution_descriptor_1x1()?,
        alpha[1],
        create_offset_descriptor(num_outputs)?,
        cudnn2::ActivationDescriptor::relu()?,
        create_tensor_descriptor(batch_size, num_outputs)?
    )
}

// -------- Graph --------

pub struct Builder {
    tensors: Arc<HashMap<String, Tensor>>,
    slots: Slots
}

impl Builder {
    pub fn new(tensors: HashMap<String, Tensor>) -> Builder {
        Builder {
            tensors: Arc::new(tensors),
            slots: Slots::new()
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
        let c_value = unsafe { Rc::new(ValueLayer::new(&handle_dnn, batch_size as i32, 3 + c_residual.len(), &self.tensors)?) };
        let c_policy = unsafe { Rc::new(PolicyLayer::new(&handle_dnn, batch_size as i32, 3 + c_residual.len(), &self.tensors)?) };

        let mut w = Workspace {
            batch_size: batch_size,
            tensors: self.tensors.clone(),
            slots: self.slots.clone(),
            num_channels: c_residual[0].num_channels,

            handle_blas: ptr::null(),
            handle_dnn: handle_dnn,

            tower_finished: cuda2::Event::new()?,

            tower_stream: cuda2::Stream::new()?,
            policy_stream: cuda2::Stream::new()?,
            value_stream: cuda2::Stream::new()?,

            c_up: c_up,
            c_value: c_value,
            c_policy: c_policy,
            c_residual: c_residual
        };

        unsafe {
            check!(cublas::cublasCreate_v2(&mut w.handle_blas))?;

            #[cfg(feature = "tensor-core")] {
                check!(cublas::cublasSetMathMode(w.handle_blas, cublas::Math::TensorOp))?;
            }
        }

        Ok(w)
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
    tensors: Arc<HashMap<String, Tensor>>,
    slots: Slots,
    num_channels: usize,

    handle_dnn: cudnn2::Handle,
    handle_blas: cublas::Handle,

    tower_finished: cuda2::Event,

    tower_stream: cuda2::Stream,
    policy_stream: cuda2::Stream,
    value_stream: cuda2::Stream,

    c_up: Rc<UpLayer>,
    c_value: Rc<ValueLayer>,
    c_policy: Rc<PolicyLayer>,
    c_residual: Vec<Rc<ResidualLayer>>
}

impl Drop for Workspace {
    fn drop(&mut self) {
        unsafe {
            cublas::cublasDestroy_v2(self.handle_blas);
        }
    }
}

struct UpLayer {
    up: cudnn2::ConvolutionBiasActivation,
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
            up: create_convolution_bias_activation_3x3(handle, n, num_channels, NUM_FEATURES as i32, &[1.0, 0.0])?
        })
    }

    unsafe fn forward<'a, T: InferenceType>(
        &self,
        workspace: &mut Workspace,
        slots: &'a SlotsGuard,
        input: &SlotGuard<'a>
    ) -> Result<SlotGuard<'a>, Error>
    {
        workspace.handle_dnn.set_stream(&workspace.tower_stream)?;

        let device_id = get_current_device()?;
        let weights = &workspace.tensors["01_upsample/conv_1:0"];
        let offset = &workspace.tensors["01_upsample/conv_1/offset:0"];

        offset.copy_to_device(device_id, *workspace.tower_stream)?;
        weights.copy_to_device(device_id, *workspace.tower_stream)?;

        // perform the forward convolution
        let workspace_1 = slots.get_slot(Slot::Workspace_1, self.up.fwd_algo_perf().memory(), *workspace.tower_stream)?;
        let output = slots.get_slot(Slot::Residual_1, size_of::<T::Tower>() * workspace.batch_size * workspace.num_channels * 361, *workspace.tower_stream)?;

        self.up.forward(
            &workspace.handle_dnn,
            **input,
            weights.get(device_id),
            *workspace_1, self.up.fwd_algo_perf().memory(),
            *output,
            offset.get(device_id),
            *output
        )?;

        Ok(output)
    }
}

struct ResidualLayer {
    conv_1: cudnn2::ConvolutionBiasActivation,
    conv_2: cudnn2::ConvolutionBiasActivation,
    gate_t: f32,

    num_channels: usize,
    count: usize,
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
            conv_1: create_convolution_bias_activation_3x3(handle, n, num_channels, num_channels, &[1.0, 0.0])?,
            conv_2: create_convolution_bias_activation_3x3(handle, n, num_channels, num_channels, &[gate_t, 1.0 - gate_t])?,
            gate_t: gate_t,
            num_channels: num_channels as usize,
            count: i,
        }))
    }

    unsafe fn forward<'a, T: InferenceType>(
        &self,
        workspace: &mut Workspace,
        slots: &'a SlotsGuard,
        input: SlotGuard<'a>
    ) -> Result<SlotGuard<'a>, Error>
    {
        workspace.handle_dnn.set_stream(&workspace.tower_stream)?;
        check!(cublas::cublasSetStream_v2(workspace.handle_blas, *workspace.tower_stream))?;

        let device_id = get_current_device()?;
        let weights_1 = &workspace.tensors[&format!("{:02}_residual/conv_1:0", self.count)];
        let weights_2 = &workspace.tensors[&format!("{:02}_residual/conv_2:0", self.count)];
        let offset_1 = &workspace.tensors[&format!("{:02}_residual/conv_1/offset:0", self.count)];
        let offset_2 = &workspace.tensors[&format!("{:02}_residual/conv_2/offset:0", self.count)];

        weights_1.copy_to_device(device_id, *workspace.tower_stream)?;
        weights_2.copy_to_device(device_id, *workspace.tower_stream)?;
        offset_1.copy_to_device(device_id, *workspace.tower_stream)?;
        if offset_2.copy_to_device(device_id, *workspace.tower_stream)? {
            check!(cudnn::cudnnScaleTensor(
                *workspace.handle_dnn,
                **self.conv_1.offset(), offset_2.get(device_id),
                &self.gate_t
            ))?;
        }

        debug_assert!(offset_1.size_in_elements == offset_2.size_in_elements);

        // perform the forward convolution (1)
        let workspace_r = slots.get_slot(Slot::Workspace_r, self.conv_1.fwd_algo_perf().memory(), *workspace.tower_stream)?;
        let residual_2_size = size_of::<T::Tower>() * workspace.batch_size * workspace.num_channels * 361;
        let residual_2 = slots.get_slot(Slot::Residual_2, residual_2_size, *workspace.tower_stream)?;

        self.conv_1.forward(
            &workspace.handle_dnn,
            *input,
            weights_1.get(device_id),
            *workspace_r, self.conv_1.fwd_algo_perf().memory(),
            *residual_2,
            offset_1.get(device_id),
            *residual_2
        )?;

        self.conv_2.forward(
            &workspace.handle_dnn,
            *residual_2,
            weights_2.get(device_id),
            *workspace_r, self.conv_2.fwd_algo_perf().memory(),
            *input,
            offset_2.get(device_id),
            *input
        )?;

        Ok(input)
    }
}

struct ValueLayer {
    conv_1: cudnn2::ConvolutionBiasActivation,
    value_2: cudnn2::TensorDescriptor,
    value_3: cudnn2::TensorDescriptor,
    bias_1: cudnn2::TensorDescriptor,
    bias_2: cudnn2::TensorDescriptor,
    relu: cudnn2::ActivationDescriptor,
    tanh: cudnn2::ActivationDescriptor,

    count: usize
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
            conv_1: create_convolution_bias_activation_1x1(handle, n, 2, num_channels, &[1.0, 0.0])?,
            value_2: create_dense_descriptor(n, 256)?,
            value_3: create_dense_descriptor(n, 1)?,
            bias_1: create_offset_descriptor(256)?,
            bias_2: create_offset_descriptor(1)?,
            relu: cudnn2::ActivationDescriptor::relu()?,
            tanh: cudnn2::ActivationDescriptor::tanh()?,
            count: i
        })
    }

    unsafe fn forward<'a, T: InferenceType>(
        &self,
        workspace: &mut Workspace,
        slots: &'a SlotsGuard,
        output_set: &OutputSet,
        output_map: &mut OutputMap<Vec<f32>>,
        input: &SlotGuard<'a>
    ) -> Result<SlotGuard<'a>, Error>
    {
        workspace.handle_dnn.set_stream(&workspace.value_stream)?;
        check!(cublas::cublasSetStream_v2(workspace.handle_blas, *workspace.value_stream))?;

        let device_id = get_current_device()?;
        let weights_1 = &workspace.tensors[&format!("{:02}v_value/conv_1:0", self.count)];
        let weights_2 = &workspace.tensors[&format!("{:02}v_value/linear_1:0", self.count)];
        let weights_3 = &workspace.tensors[&format!("{:02}v_value/linear_2:0", self.count)];
        let offset_1 = &workspace.tensors[&format!("{:02}v_value/conv_1/offset:0", self.count)];
        let offset_2 = &workspace.tensors[&format!("{:02}v_value/linear_1/offset:0", self.count)];
        let offset_3 = &workspace.tensors[&format!("{:02}v_value/linear_2/offset:0", self.count)];

        weights_1.copy_to_device(device_id, *workspace.value_stream)?;
        weights_2.copy_to_device(device_id, *workspace.value_stream)?;
        weights_3.copy_to_device(device_id, *workspace.value_stream)?;
        offset_1.copy_to_device(device_id, *workspace.value_stream)?;
        offset_2.copy_to_device(device_id, *workspace.value_stream)?;
        offset_3.copy_to_device(device_id, *workspace.value_stream)?;

        // perform the forward convolution
        let workspace_v = slots.get_slot(Slot::Workspace_v, self.conv_1.fwd_algo_perf().memory(), *workspace.value_stream)?;
        let value_1 = slots.get_slot(Slot::Value_1, size_of::<T::Output>() * workspace.batch_size * 722, *workspace.value_stream)?;

        self.conv_1.forward(
            &workspace.handle_dnn,
            **input,
            weights_1.get(device_id),
            *workspace_v, self.conv_1.fwd_algo_perf().memory(),
            *value_1,
            offset_1.get(device_id),
            *value_1
        )?;

        load_output::<T::Output>(output_set, output_map, Output::ValueDown, *value_1, workspace.batch_size * 722, *workspace.value_stream)?;

        // perform the feed-forward linear layer (relu)
        let value_2 = slots.get_slot(Slot::Value_2, size_of::<T::Output>() * workspace.batch_size * 256, *workspace.value_stream)?;
        let value_3 = slots.get_slot(Slot::Value_3, size_of::<T::Output>() * workspace.batch_size * 1, *workspace.value_stream)?;

        check!(cublas::cublasGemmEx(
            workspace.handle_blas,
            cublas::Operation::N,
            cublas::Operation::N,
            256, workspace.batch_size as i32, 722,  // output, batch_size, input
            &ONE as *const f32 as *const c_void,
            weights_2.get(device_id), cuda::DataType::R16F, 256,  // input_2
            *value_1, cuda::DataType::R16F, 722,  // input_1
            &ZERO as *const f32 as *const c_void,
            *value_2, cuda::DataType::R16F, 256,  // output
            cuda::DataType::R32F, cublas::GemmAlgo::DfaltTensorOp
        ))?;

        check!(cudnn::cudnnAddTensor(
            *workspace.handle_dnn,
            &ONE, *self.bias_1, offset_2.get(device_id),
            &ONE, *self.value_2, *value_2
        ))?;

        check!(cudnn::cudnnActivationForward(
            *workspace.handle_dnn,
            *self.relu,
            &ONE, *self.value_2, *value_2,  // input
            &ZERO, *self.value_2, *value_2,  // output
        ))?;

        load_output::<T::Output>(output_set, output_map, Output::ValueGemm, *value_2, workspace.batch_size * 256, *workspace.value_stream)?;

        // perform the feed-forward linear layer (tanh)
        check!(cublas::cublasGemmEx(
            workspace.handle_blas,
            cublas::Operation::N,
            cublas::Operation::N,
            1, workspace.batch_size as i32, 256,  // output, batch_size, input
            &ONE as *const f32 as *const c_void,
            weights_3.get(device_id), cuda::DataType::R16F, 1,  // input_2
            *value_2, cuda::DataType::R16F, 256,  // input_1
            &ZERO as *const f32 as *const c_void,
            *value_3, cuda::DataType::R16F, 1,  // output
            cuda::DataType::R32F, cublas::GemmAlgo::DfaltTensorOp
        ))?;

        check!(cudnn::cudnnAddTensor(
            *workspace.handle_dnn,
            &ONE, *self.bias_2, offset_3.get(device_id),
            &ONE, *self.value_3, *value_3
        ))?;

        check!(cudnn::cudnnActivationForward(
            *workspace.handle_dnn,
            *self.tanh,
            &ONE, *self.value_3, *value_3,  // input
            &ZERO, *self.value_3, *value_3,  // output
        ))?;

        Ok(value_3)
    }
}

struct PolicyLayer {
    conv_1: cudnn2::ConvolutionBiasActivation,
    policy_2: cudnn2::TensorDescriptor,
    offset_2: cudnn2::TensorDescriptor,
    count: usize
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

        Ok(PolicyLayer {
            conv_1: create_convolution_bias_activation_1x1(handle, n, 4, num_channels, &[1.0, 0.0])?,
            policy_2: create_dense_descriptor(n, 362)?,
            offset_2: create_offset_descriptor(362)?,
            count: i
        })
    }

    unsafe fn forward<'a, T: InferenceType>(
        &self,
        workspace: &mut Workspace,
        slots: &'a SlotsGuard,
        output_set: &OutputSet,
        output_map: &mut OutputMap<Vec<f32>>,
        input: &SlotGuard<'a>
    ) -> Result<SlotGuard<'a>, Error>
    {
        workspace.handle_dnn.set_stream(&workspace.policy_stream)?;
        check!(cublas::cublasSetStream_v2(workspace.handle_blas, *workspace.policy_stream))?;

        let device_id = get_current_device()?;
        let weights_1 = &workspace.tensors[&format!("{:02}p_policy/conv_1:0", self.count)];
        let weights_2 = &workspace.tensors[&format!("{:02}p_policy/linear_1:0", self.count)];
        let offset_1 = &workspace.tensors[&format!("{:02}p_policy/conv_1/offset:0", self.count)];
        let offset_2 = &workspace.tensors[&format!("{:02}p_policy/linear_1/offset:0", self.count)];

        offset_1.copy_to_device(device_id, *workspace.policy_stream)?;
        offset_2.copy_to_device(device_id, *workspace.policy_stream)?;
        weights_1.copy_to_device(device_id, *workspace.policy_stream)?;
        weights_2.copy_to_device(device_id, *workspace.policy_stream)?;

        // perform the forward convolution
        let workspace_p = slots.get_slot(Slot::Workspace_p, self.conv_1.fwd_algo_perf().memory(), *workspace.policy_stream)?;
        let policy_1 = slots.get_slot(Slot::Policy_1, size_of::<T::Output>() * workspace.batch_size * 1444, *workspace.policy_stream)?;

        self.conv_1.forward(
            &workspace.handle_dnn,
            **input,
            weights_1.get(device_id),
            *workspace_p, self.conv_1.fwd_algo_perf().memory(),
            *policy_1,
            offset_1.get(device_id),
            *policy_1
        )?;

        load_output::<T::Output>(output_set, output_map, Output::PolicyDown, *policy_1, workspace.batch_size * 1444, *workspace.policy_stream)?;

        // perform the feed-forward linear layers
        let policy_2 = slots.get_slot(Slot::Policy_2, size_of::<T::Output>() * workspace.batch_size * 362, *workspace.policy_stream)?;
        let policy_3 = slots.get_slot(Slot::Policy_3, size_of::<T::Output>() * workspace.batch_size * 362, *workspace.policy_stream)?;

        check!(cublas::cublasGemmEx(
            workspace.handle_blas,
            cublas::Operation::N,
            cublas::Operation::N,
            362, workspace.batch_size as i32, 1444,  // output, batch_size, input
            &ONE as *const f32 as *const c_void,
            weights_2.get(device_id), cuda::DataType::R16F, 362,  // input_2
            *policy_1, cuda::DataType::R16F, 1444,  // input_1
            &ZERO as *const f32 as *const c_void,
            *policy_2, cuda::DataType::R16F, 362,  // output
            cuda::DataType::R32F, cublas::GemmAlgo::DfaltTensorOp
        ))?;

        // apply the softmax temperature at the _add tensor_ layer since the cuDNN
        // _softmax_ primitive does not support it directly.
        lazy_static! {
            static ref TAU: f32 = 1.0 / *config::SOFTMAX_TEMPERATURE;
        }

        check!(cudnn::cudnnAddTensor(
            *workspace.handle_dnn,
            &*TAU, *self.offset_2, offset_2.get(device_id),
            &*TAU, *self.policy_2, *policy_2
        ))?;

        // softmax activation
        check!(cudnn::cudnnSoftmaxForward(
            *workspace.handle_dnn,
            cudnn::SoftmaxAlgorithm::Accurate,
            cudnn::SoftmaxMode::Instance,
            &ONE, *self.policy_2, *policy_2,  // input
            &ZERO, *self.policy_2, *policy_3,  // output
        ))?;

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

    let slots = workspace.slots.lock()?;
    let mut map = OutputMap::default();

    unsafe {
        workspace.handle_dnn.set_stream(&workspace.tower_stream)?;

        // copy all of the input features into a temporary workspace
        let input = slots.get_slot(Slot::Input, size_of::<T>() * features.len(), *workspace.tower_stream)?;
        let image_size = 361 * workspace.num_channels;

        check!(cuda::cudaMemcpyAsync(
            *input,
            features.as_ptr() as *const c_void,
            size_of::<T>() * features.len(),
            cuda::MemcpyKind::HostToDevice,
            *workspace.tower_stream
        ))?;

        // Upsample 32 -> 128 channels
        let mut residual_1 = workspace.c_up.clone().forward::<T>(workspace, &slots, &input)?;

        load_output::<T::Tower>(&outputs, &mut map, Output::Upsample, *residual_1, workspace.batch_size * image_size, *workspace.tower_stream)?;

        // residual blocks
        let num_residual = workspace.c_residual.len();

        for i in 0..num_residual {
            let residual = workspace.c_residual[i].clone();
            let output = ::std::mem::transmute(Output::Residual_00 as u8 + i as u8);

            residual_1 = residual.forward::<T>(workspace, &slots, residual_1)?;
            load_output::<T::Tower>(&outputs, &mut map, output, *residual_1, workspace.batch_size * image_size, *workspace.tower_stream)?;
        }

        workspace.tower_finished.record(&workspace.tower_stream)?;
        workspace.value_stream.wait_event(&workspace.tower_finished)?;
        workspace.policy_stream.wait_event(&workspace.tower_finished)?;

        // run the value and policy head, then wait for them to finish (if
        // they are requested)
        let value = workspace.c_value.clone().forward::<T>(workspace, &slots, &outputs, &mut map, &residual_1)?;
        let policy = workspace.c_policy.clone().forward::<T>(workspace, &slots, &outputs, &mut map, &residual_1)?;

        load_output::<T::Output>(&outputs, &mut map, Output::Value, *value, workspace.batch_size, *workspace.value_stream)?;
        load_output::<T::Output>(&outputs, &mut map, Output::Policy, *policy, workspace.batch_size * 362, *workspace.policy_stream)?;
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
