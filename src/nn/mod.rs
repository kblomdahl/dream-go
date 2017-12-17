// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use libc::{c_void};

#[macro_use] mod ffi;
mod loader;
mod network;
mod workspace;

use self::ffi::cublas::*;
use self::ffi::cuda::*;
use self::ffi::cudnn::*;
pub use self::network::{Network, WorkspaceGuard};
pub use self::workspace::Workspace;
use util::f16::*;

/// Returns the value and policy tensors obtained from a forward pass
/// through the neural network.
///
/// # Arguments
///
/// * `ws` - the workspace for the current thread
/// * `features` - the input features
///
pub fn forward<T: From<f32> + Clone>(
    w: &mut Workspace,
    features: &Vec<Box<[T]>>
) -> (Vec<T>, Vec<Box<[T]>>)
{
    assert_eq!(w.batch_size, features.len());
    assert_eq!(::std::mem::size_of::<T>(), w.shared.data_type.size());

    let epsilon: f64 = 0.001;  // tensorflow default
    let c_0 = 0.0f32;
    let c_1 = 1.0f32;
    let ch_0 = f16::from(c_0);
    let ch_1 = f16::from(c_1);

    let mut softmax = vec! [vec! [T::from(0.0f32); 362]; w.batch_size];
    let mut value = vec! [T::from(0.0f32); w.batch_size];

    unsafe {
        check!(cudnnSetStream(w.handle_dnn, w.tower_s));

        for (i, ref feature) in features.iter().enumerate() {
            assert_eq!(feature.len(), 12274);
            assert_eq!(1, ::std::mem::size_of::<c_void>());

            let element_size = 12274 * w.shared.data_type.size();

            check!(cudaMemcpyAsync(
                w.input.offset((i * element_size) as isize),
                feature.as_ptr() as *const c_void,
                element_size,
                MemcpyKind::HostToDevice,
                w.tower_s
            ));
        }

        // up-sample the input features to the 256-wide internal representation
        check!(cudnnConvolutionForward(
            w.handle_dnn,
            &c_1,  // alpha
            w.input_t, w.input,  // input
            w.shared.up_f, w.shared.weights["01_upsample/weights:0"],  // weights
            w.shared.conv2d_3,  // convolution
            w.shared.get_convolution_algo(3),  // algo
            w.scratch_1, w.scratch_size,  // workspace
            &c_0,  // beta
            w.residual_t, w.residual_1,  // output
        ), w, "01_upsample/up", w.residual_1, w.batch_size, NUM_FEATURES, 361);

        check!(cudnnBatchNormalizationForwardInference(
            w.handle_dnn,
            BatchNormMode::Spatial,
            &c_1,  // alpha
            &c_0,  // beta
            w.residual_t, w.residual_1,  // input
            w.residual_t, w.residual_2,  // output
            w.residual_bn_t,
            w.shared.ones, w.shared.zeros,  // scale, bias
            w.shared.weights["01_upsample/mean:0"],
            w.shared.weights["01_upsample/variance:0"],
            epsilon
        ), w, "01_upsample/up_bn", w.residual_2, w.batch_size, NUM_FEATURES, 361);

        check!(cudnnActivationForward(
            w.handle_dnn,
            w.shared.relu,
            &c_1,  // alpha
            w.residual_t, w.residual_2,  // input
            &c_0,  // beta
            w.residual_t, w.residual_1,  // output
        ), w, "01_upsample/up_relu", w.residual_1, w.batch_size, NUM_FEATURES, 361);

        // apply all of the residual blocks
        for i in 2..21 {
            check!(cudnnConvolutionForward(
                w.handle_dnn,
                &c_1,  // alpha
                w.residual_t, w.residual_1,  // input
                w.shared.residual_f, w.shared.weights[&format!("{:02}_residual/weights_1:0", i)],  // weights
                w.shared.conv2d_3,  // convolution
                w.shared.get_convolution_algo(3),  // algo
                w.scratch_1, w.scratch_size,  // workspace
                &c_0,  // beta
                w.residual_t, w.residual_2,  // output
            ), w, &format!("{:02}_residual/conv_1", i), w.residual_2, w.batch_size, NUM_FEATURES, 361);

            check!(cudnnBatchNormalizationForwardInference(
                w.handle_dnn,
                BatchNormMode::Spatial,
                &c_1,  // alpha
                &c_0,  // beta
                w.residual_t, w.residual_2,  // input
                w.residual_t, w.residual_3,  // output
                w.residual_bn_t,
                w.shared.ones, w.shared.zeros,  // scale, bias
                w.shared.weights[&format!("{:02}_residual/mean_1:0", i)],
                w.shared.weights[&format!("{:02}_residual/variance_1:0", i)],
                epsilon
            ), w, &format!("{:02}_residual/conv_bn_1", i), w.residual_3, w.batch_size, NUM_FEATURES, 361);

            check!(cudnnActivationForward(
                w.handle_dnn,
                w.shared.relu,
                &c_1,  // alpha
                w.residual_t, w.residual_3,  // input
                &c_0,  // beta
                w.residual_t, w.residual_3,  // output
            ), w, &format!("{:02}_residual/conv_relu_1", i), w.residual_3, w.batch_size, NUM_FEATURES, 361);

            check!(cudnnConvolutionForward(
                w.handle_dnn,
                &c_1,  // alpha
                w.residual_t, w.residual_3,  // input
                w.shared.residual_f, w.shared.weights[&format!("{:02}_residual/weights_2:0", i)],  // weights
                w.shared.conv2d_3,  // convolution
                w.shared.get_convolution_algo(3),  // algo
                w.scratch_1, w.scratch_size,  // workspace
                &c_0,  // beta
                w.residual_t, w.residual_2,  // output
            ), w, &format!("{:02}_residual/conv_2", i), w.residual_2, w.batch_size, NUM_FEATURES, 361);

            check!(cudnnBatchNormalizationForwardInference(
                w.handle_dnn,
                BatchNormMode::Spatial,
                &c_1,  // alpha
                &c_1,  // beta
                w.residual_t, w.residual_2,  // input
                w.residual_t, w.residual_1,  // output
                w.residual_bn_t,
                w.shared.ones, w.shared.zeros,  // scale, bias
                w.shared.weights[&format!("{:02}_residual/mean_2:0", i)],
                w.shared.weights[&format!("{:02}_residual/variance_2:0", i)],
                epsilon
            ), w, &format!("{:02}_residual/conv_bn_2", i), w.residual_1, w.batch_size, NUM_FEATURES, 361);

            check!(cudnnActivationForward(
                w.handle_dnn,
                w.shared.relu,
                &c_1,  // alpha
                w.residual_t, w.residual_1,  // input
                &c_0,  // beta
                w.residual_t, w.residual_1,  // output
            ), w, &format!("{:02}_residual/conv_relu_2", i), w.residual_1, w.batch_size, NUM_FEATURES, 361);
        }

        check!(cudaEventRecord(w.tower_e, w.tower_s));
        check!(cudaStreamWaitEvent(w.policy_s, w.tower_e, 0));
        check!(cudaStreamWaitEvent(w.value_s, w.tower_e, 0));

        // policy head (21p_policy)
        check!(cudnnSetStream(w.handle_dnn, w.policy_s));
        check!(cublasSetStream_v2(w.handle_blas, w.policy_s));
        check!(cudnnConvolutionForward(
            w.handle_dnn,
            &c_1,  // alpha
            w.residual_t, w.residual_1,  // input
            w.shared.policy_f, w.shared.weights["21p_policy/downsample:0"],  // weights
            w.shared.conv2d_1,  // convolution
            w.shared.get_convolution_algo(1),  // algo
            w.scratch_1, w.scratch_size,  // workspace
            &c_0,  // beta
            w.policy_t, w.policy_1,  // output
        ), w, "21p_policy/down", w.policy_1, w.batch_size, 2, 361);

        check!(cudnnBatchNormalizationForwardInference(
            w.handle_dnn,
            BatchNormMode::Spatial,
            &c_1,  // alpha
            &c_0,  // beta
            w.policy_t, w.policy_1,  // input
            w.policy_t, w.policy_2,  // output
            w.policy_bn_t,
            w.shared.ones, w.shared.zeros,  // scale, bias
            w.shared.weights["21p_policy/mean:0"],
            w.shared.weights["21p_policy/variance:0"],
            epsilon
        ), w, "21p_policy/down_bn", w.policy_2, w.batch_size, 2, 361);

        check!(cudnnActivationForward(
            w.handle_dnn,
            w.shared.relu,
            &c_1,  // alpha
            w.policy_t, w.policy_2,  // input
            &c_0,  // beta
            w.policy_t, w.policy_2,  // output
        ), w, "21p_policy/down_relu", w.policy_2, w.batch_size, 2, 361);

        if w.shared.is_half() {
            check!(cublasHgemm(
                w.handle_blas,
                Operation::N,
                Operation::N,
                362, w.batch_size as i32, 722, // output_dims, batch_size, input_dims
                &ch_1,  // alpha
                w.shared.weights["21p_policy/weights:0"], 362,  // input_2
                w.policy_2, 722,  // input_1
                &ch_0,  // beta
                w.policy_1, 362  // output
            ), w, "21p_policy/ff", w.policy_1, w.batch_size, 362);
        } else {
            check!(cublasSgemm_v2(
                w.handle_blas,
                Operation::N,
                Operation::N,
                362, w.batch_size as i32, 722, // output_dims, batch_size, input_dims
                &c_1,  // alpha
                w.shared.weights["21p_policy/weights:0"], 362,  // input_2
                w.policy_2, 722,  // input_1
                &c_0,  // beta
                w.policy_1, 362  // output
            ), w, "21p_policy/ff", w.policy_1, w.batch_size, 362);
        }

        check!(cudnnAddTensor(
            w.handle_dnn,
            &c_1,  // alpha
            w.shared.policy_bias_t, w.shared.weights["21p_policy/bias:0"],  // bias
            &c_1,  // beta
            w.policy_softmax_t, w.policy_1  // input and output
        ), w, "21p_policy/bias", w.policy_1, w.batch_size, 362);

        check!(cudnnSoftmaxForward(
            w.handle_dnn,
            SoftmaxAlgorithm::Accurate,
            SoftmaxMode::Instance,
            &c_1,  // alpha
            w.policy_softmax_t, w.policy_1,  // input
            &c_0,  // beta
            w.policy_softmax_t, w.policy_2  // output
        ), w, "21p_policy/softmax", w.policy_2, w.batch_size, 362);

        for i in 0..w.batch_size {
            let element_size = 362 * w.shared.data_type.size();

            check!(cudaMemcpyAsync(
                softmax[i].as_mut_ptr() as *mut c_void,
                w.policy_2.offset((i * element_size) as isize),
                element_size,
                MemcpyKind::DeviceToHost,
                w.policy_s
            ));
        }

        // value head (21v_value)
        check!(cudnnSetStream(w.handle_dnn, w.value_s));
        check!(cublasSetStream_v2(w.handle_blas, w.value_s));
        check!(cudnnConvolutionForward(
            w.handle_dnn,
            &c_1,  // alpha
            w.residual_t, w.residual_1,  // input
            w.shared.value_f, w.shared.weights["21v_value/downsample:0"],  // weights
            w.shared.conv2d_1,  // convolution
            w.shared.get_convolution_algo(1),  // algo
            w.scratch_2, w.scratch_size,  // workspace
            &c_0,  // beta
            w.value_t, w.value_1  // output
        ), w, "21v_value/down", w.value_1, w.batch_size, 361);

        check!(cudnnBatchNormalizationForwardInference(
            w.handle_dnn,
            BatchNormMode::Spatial,
            &c_1,  // alpha
            &c_0,  // beta
            w.value_t, w.value_1,  // input
            w.value_t, w.value_2,  // output
            w.value_bn_t,
            w.shared.ones, w.shared.zeros,  // scale, bias
            w.shared.weights["21v_value/mean:0"],
            w.shared.weights["21v_value/variance:0"],
            epsilon
        ), w, "21v_value/down_bn", w.value_2, w.batch_size, 361);

        check!(cudnnActivationForward(
            w.handle_dnn,
            w.shared.relu,
            &c_1,  // alpha
            w.value_t, w.value_2,  // input
            &c_0,  // beta
            w.value_t, w.value_2,  // output
        ), w, "21v_value/down_relu", w.value_2, w.batch_size, 361);

        if w.shared.is_half() {
            check!(cublasHgemm(
                w.handle_blas,
                Operation::N,
                Operation::N,
                256, w.batch_size as i32, 361,  // output_dims, batch_size, input_dims
                &ch_1,  // alpha
                w.shared.weights["21v_value/weights_1:0"], 256,  // input_2
                w.value_2, 361,  // input_1
                &ch_0,  // beta
                w.value_1, 256  // output
            ), w, "21v_value/ff_256", w.value_1, w.batch_size, 256);
        } else {
            check!(cublasSgemm_v2(
                w.handle_blas,
                Operation::N,
                Operation::N,
                256, w.batch_size as i32, 361,  // output_dims, batch_size, input_dims
                &c_1,  // alpha
                w.shared.weights["21v_value/weights_1:0"], 256,  // input_2
                w.value_2, 361,  // input_1
                &c_0,  // beta
                w.value_1, 256  // output
            ), w, "21v_value/ff_256", w.value_1, w.batch_size, 256);
        }

        check!(cudnnAddTensor(
            w.handle_dnn,
            &c_1,  // alpha
            w.shared.value_256_bias_t, w.shared.weights["21v_value/bias_1:0"],  // bias
            &c_1,  // beta
            w.value_256_t, w.value_1  // input and output
        ), w, "21v_value/ff_bias_256", w.value_1, w.batch_size, 256);

        check!(cudnnActivationForward(
            w.handle_dnn,
            w.shared.relu,
            &c_1,  // alpha
            w.value_256_t, w.value_1,  // input
            &c_0,  // beta
            w.value_256_t, w.value_1,  // output
        ), w, "21v_value/ff_relu_256", w.value_1, w.batch_size, 256);

        if w.shared.is_half() {
            check!(cublasHgemm(
                w.handle_blas,
                Operation::N,
                Operation::N,
                1, w.batch_size as i32, 256,  // output_dims, batch_size, input_dims
                &ch_1,  // alpha
                w.shared.weights["21v_value/weights_2:0"], 1,  // input_2
                w.value_1, 256,  // input_1
                &ch_0,  // beta
                w.value_2, 1  // output
            ), w, "21v_value/ff_1", w.value_2, w.batch_size, 1);
        } else {
            check!(cublasSgemm_v2(
                w.handle_blas,
                Operation::N,
                Operation::N,
                1, w.batch_size as i32, 256,  // output_dims, batch_size, input_dims
                &c_1,  // alpha
                w.shared.weights["21v_value/weights_2:0"], 1,  // input_2
                w.value_1, 256,  // input_1
                &c_0,  // beta
                w.value_2, 1  // output
            ), w, "21v_value/ff_1", w.value_2, w.batch_size, 1);
        }

        check!(cudnnAddTensor(
            w.handle_dnn,
            &c_1,  // alpha
            w.shared.value_1_bias_t, w.shared.weights["21v_value/bias_2:0"],  // bias
            &c_1,  // beta
            w.value_1_t, w.value_2  // input and output
        ), w, "21v_value/ff_bias_1", w.value_2, w.batch_size, 1);

        check!(cudnnActivationForward(
            w.handle_dnn,
            w.shared.tanh,
            &c_1,  // alpha
            w.value_1_t, w.value_2,  // input
            &c_0,  // beta
            w.value_1_t, w.value_2,  // output
        ), w, "21v_value/ff_tanh_2", w.value_2, w.batch_size, 1);

        check!(cudaMemcpyAsync(
            value.as_mut_ptr() as *mut c_void,
            w.value_2,
            w.batch_size * w.shared.data_type.size(),
            MemcpyKind::DeviceToHost,
            w.value_s
        ));

        // wait for both the value and policy head to finish
        check!(cudaStreamSynchronize(w.policy_s));
        check!(cudaStreamSynchronize(w.value_s));
    }

    (value, softmax.into_iter().map(|s| s.into_boxed_slice()).collect())
}

#[cfg(test)]
mod tests {
    use test::Bencher;
    use rand::{Rng, thread_rng};
    use nn::*;

    #[allow(dead_code)]
    fn bench_batch_size(b: &mut Bencher, batch_size: usize) {
        let network = Network::new().unwrap();
        let mut workspace = network.get_workspace(batch_size);

        // allocate a feature vector filled with random ones and zeros
        if network.is_half() {
            let features = (0..batch_size).map(|_| {
                let mut input = vec! [f16::from(0.0); 12274];

                for b in input.iter_mut() {
                    *b = f16::from(if thread_rng().next_f32() < 0.2 { 1.0 } else { 0.0 });
                }

                input.into_boxed_slice()
            }).collect();

            b.iter(move || {
                forward(&mut workspace, &features)
            });
        } else {
            let features = (0..batch_size).map(|_| {
                let mut input = vec! [0.0f32; 12274];

                for b in input.iter_mut() {
                    *b = if thread_rng().next_f32() < 0.2 { 1.0 } else { 0.0 };
                }

                input.into_boxed_slice()
            }).collect();

            b.iter(move || {
                forward(&mut workspace, &features)
            });
        }
    }

    #[bench] fn batch_size_01(b: &mut Bencher) { bench_batch_size(b,  1); }
    #[bench] fn batch_size_02(b: &mut Bencher) { bench_batch_size(b,  2); }
    #[bench] fn batch_size_04(b: &mut Bencher) { bench_batch_size(b,  4); }
    #[bench] fn batch_size_08(b: &mut Bencher) { bench_batch_size(b,  8); }
    #[bench] fn batch_size_16(b: &mut Bencher) { bench_batch_size(b, 16); }
    #[bench] fn batch_size_32(b: &mut Bencher) { bench_batch_size(b, 32); }
}