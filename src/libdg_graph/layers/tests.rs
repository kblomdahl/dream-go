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

use dg_cuda::cudnn::{cudnnDataType_t, Handle, Tensor};
use dg_cuda::{Stream, Ptr, copy_nonoverlapping, cudaMemcpyKind_t};
use rand::{thread_rng, Rng};
use layer::Layer;
use graph_def::{LayerDef, VariableDef};
use std::fmt::Display;

#[cfg(test)]
pub fn allocate_tensor<T: From<f32>>(
    var_def: &VariableDef,
    data_type: cudnnDataType_t,
    stream: &Stream
) -> (Tensor, Vec<T>, Ptr)
{
    let shape: Vec<usize> = var_def.shape.iter().map(|&x| x as usize).collect();
    let tensor = Tensor::from_nhwc(data_type, &shape)
        .expect("Could not create input Tensor");

    let host = (0..tensor.len().expect("Could not get length of input tensor"))
        .map(|_i| T::from(1.0 - 2.0 * thread_rng().gen::<f32>()))
        .collect();
    let device = Ptr::from_vec(
        &host,
        &stream
    ).expect("Could not allocate input buffer");

    (tensor, host, device)
}

#[cfg(test)]
pub fn run_layer<T: Sized + Copy + Default + From<f32>, L: Layer>(
    layer_def: &LayerDef,
    layer: &L,
    data_type: cudnnDataType_t
) -> (Vec<Vec<T>>, Vec<Vec<T>>)
{
    let mut handle = Handle::new().expect("Could not create handle");
    let stream = Stream::new().expect("Could not create stream");
    handle.set_stream(&stream).expect("Could not set cuDNN stream");

    // allocate the I/O buffers that will hold both the input and output
    let inputs: Vec<_> = layer_def.input.iter().map(|input_def| {
        allocate_tensor::<T>(input_def, data_type, &stream)
    }).collect();

    let mut outputs: Vec<_> = layer_def.output.iter().map(|output_def| {
        allocate_tensor::<T>(output_def, data_type, &stream)
    }).collect();

    let prepared_layer = layer.prepare(
        &handle,
        &inputs.iter().map(|(a, _b, _c)| a).collect::<Vec<_>>(),
        &outputs.iter().map(|(a, _b, _c)| a).collect::<Vec<_>>(),
    ).expect("Could not prepare layer for execution");

    // execute the actual layer with the desired workspace (or null)
    let workspace = if prepared_layer.size_in_bytes() > 0 {
        Ptr::new(
            prepared_layer.size_in_bytes()
        ).expect("Could not allocate workspace")
    } else {
        Ptr::null()
    };

    prepared_layer.forward(
        &handle,
        &inputs.iter().map(|(a, _b, c)| (a, c.as_ptr())).collect::<Vec<_>>(),
        &outputs.iter().map(|(a, _b, c)| (a, c.as_mut_ptr())).collect::<Vec<_>>(),
        workspace.as_mut_ptr()
    ).expect("Could not execute forward phase");

    // copy the output to the host so that we can compare the result with the expect result
    // on the CPU
    for (_output, output_host, output_device) in outputs.iter_mut() {
        copy_nonoverlapping(
            output_device.as_ptr() as *const _,
            output_host.as_mut_ptr(),
            output_host.len(),
            cudaMemcpyKind_t::DeviceToHost,
            &stream
        ).expect("Could not copy output to the host buffer");
    }
    stream.synchronize().expect("Could not synchronize stream");

    (
        inputs.into_iter().map(|(_a, b, _c)| b).collect(),
        outputs.into_iter().map(|(_a, b, _c)| b).collect(),
    )
}

#[cfg(test)]
pub fn assert_approx_eq<T: Copy + Display>(actual: T, expected: T) where f32: From<T> {
    assert_approx_eq_prec(actual, expected, 0.001);
}

#[cfg(test)]
pub fn assert_approx_eq_prec<T: Copy + Display>(actual: T, expected: T, precision: f32) where f32: From<T> {
    let diff = (f32::from(actual) - f32::from(expected)).abs();

    assert!(diff <= precision, "Actual: {}, Expected: {}, Diff: {}", actual, expected, diff);
}