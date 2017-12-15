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

pub mod cublas;
pub mod cuda;
pub mod cudnn;

macro_rules! check {
    ($status:expr) => ({
        let err = $status;

        assert!(err.is_ok(), "cuda call failed -- {:?}", err);
    });

    ($status:expr, $workspace:expr, $name:expr, $result:expr, $batch_size:expr, $n:expr) => ({
        check!($status);

        #[cfg(feature = "trace-cuda")]
        {
            // copy the memory from the device back to the host so
            // that we can look at it
            let mut vec = vec! [[0.0f32; $n]; $batch_size];

            check!(cudaDeviceSynchronize());

            if $workspace.network.is_half() {
                let mut other = vec! [[f16::default(); $n]; $batch_size];

                check!(cudaMemcpy(
                    other.as_mut_ptr() as *mut c_void,
                    $result,
                    $workspace.network.data_type.size() * $batch_size * $n,
                    MemcpyKind::DeviceToHost
                ));

                for i in 0..$batch_size {
                    for j in 0..$n {
                        vec[i][j] = f32::from(other[i][j]);
                    }
                }
            } else {
                check!(cudaMemcpy(
                    vec.as_mut_ptr() as *mut c_void,
                    $result,
                    $workspace.network.data_type.size() * $batch_size * $n,
                    MemcpyKind::DeviceToHost
                ));
            }

            // pretty-print the array and then output the debugging
            // information
            let mut s = String::new();
            let mut sum = 0.0f32;

            for i in 0..$batch_size {
                if i > 0 { s += ", "; }
                s += "[";

                if $n < 8 {
                    for j in 0..$n {
                        if j > 0 { s += ", "; }

                        sum += vec[i][j];
                        s += &format!("{:.4}", vec[i][j]);
                    }
                } else {
                    for j in 0..$n {
                        sum += vec[i][j];
                    }

                    for j in 0..8 {
                        s += &format!("{:.4}, ", vec[i][j]);
                    }

                    s += "...";
                }

                s += "]";
            }

            println!("sum(`{}`) == {}", $name, sum);
            println!("eval(`{}`) == [{}]", $name, s);
        }
    });

    ($status:expr, $workspace:expr, $name:expr, $result:expr, $batch_size:expr, $m:expr, $n:expr) => ({
        check!($status, $workspace, $name, $result, $batch_size, $m * $n);
    })
}

#[cfg(test)]
mod type_bench;

#[cfg(test)]
mod tests {
    use libc::{c_void};
    use std::ptr;

    use nn::ffi::*;
    use util::f16::*;

    #[test]
    fn sgemm() {
        let mut handle: cublas::Handle = ptr::null_mut();
        let c_0 = 0.0f32;
        let c_1 = 1.0f32;

        unsafe {
            let a = [  // 3x2
                1.0f32, 2.0f32,
                3.0f32, 4.0f32,
                5.0f32, 6.0f32
            ];
            let b = [  // 2x3
                1.0f32, 2.0f32, 3.0f32,
                4.0f32, 5.0f32, 6.0f32
            ];
            let c = [  // 3x3
                0.0f32, 0.0f32, 0.0f32,
                0.0f32, 0.0f32, 0.0f32,
                0.0f32, 0.0f32, 0.0f32
            ];

            // C = A * B
            let mut a_ =  ptr::null_mut();
            let mut b_ =  ptr::null_mut();
            let mut c_ =  ptr::null_mut();

            assert_eq!(cuda::cudaMalloc(&mut a_, 24), cuda::Error::Success);
            assert_eq!(cuda::cudaMalloc(&mut b_, 24), cuda::Error::Success);
            assert_eq!(cuda::cudaMalloc(&mut c_, 36), cuda::Error::Success);
            assert_eq!(cuda::cudaMemcpy(
                a_,
                a.as_ptr() as *const c_void,
                24,
                cuda::MemcpyKind::HostToDevice
            ), cuda::Error::Success);
            assert_eq!(cuda::cudaMemcpy(
                b_,
                b.as_ptr() as *const c_void,
                24,
                cuda::MemcpyKind::HostToDevice
            ), cuda::Error::Success);

            assert_eq!(cublas::cublasCreate_v2(&mut handle), cublas::Status::Success);
            assert_eq!(cublas::cublasSgemm_v2(
                handle,
                cublas::Operation::N,
                cublas::Operation::N,
                3, 3, 2,
                &c_1,
                b_, 3,
                a_, 2,
                &c_0,
                c_, 3
            ), cublas::Status::Success);
            assert_eq!(cublas::cublasDestroy_v2(handle), cublas::Status::Success);

            // check the results
            assert_eq!(cuda::cudaMemcpy(
                c.as_ptr() as *mut c_void,
                c_,
                36,
                cuda::MemcpyKind::DeviceToHost
            ), cuda::Error::Success);

            assert_eq!(c, [
                9.0f32, 12.0f32, 15.0f32,
                19.0f32, 26.0f32, 33.0f32,
                29.0f32, 40.0f32, 51.0f32
            ])
        }
    }

    #[test]
    fn hgemm() {
        let mut handle: cublas::Handle = ptr::null_mut();
        let c_0 = f16::from(0.0f32);
        let c_1 = f16::from(1.0f32);

        unsafe {
            let a = [  // 3x2
                f16::from(1.0f32), f16::from(2.0f32),
                f16::from(3.0f32), f16::from(4.0f32),
                f16::from(5.0f32), f16::from(6.0f32)
            ];
            let b = [  // 2x3
                f16::from(1.0f32), f16::from(2.0f32), f16::from(3.0f32),
                f16::from(4.0f32), f16::from(5.0f32), f16::from(6.0f32)
            ];
            let c = [  // 3x3
                f16::from(0.0f32), f16::from(0.0f32), f16::from(0.0f32),
                f16::from(0.0f32), f16::from(0.0f32), f16::from(0.0f32),
                f16::from(0.0f32), f16::from(0.0f32), f16::from(0.0f32)
            ];

            // C = A * B
            let mut a_ =  ptr::null_mut();
            let mut b_ =  ptr::null_mut();
            let mut c_ =  ptr::null_mut();

            assert_eq!(cuda::cudaMalloc(&mut a_, 12), cuda::Error::Success);
            assert_eq!(cuda::cudaMalloc(&mut b_, 12), cuda::Error::Success);
            assert_eq!(cuda::cudaMalloc(&mut c_, 18), cuda::Error::Success);
            assert_eq!(cuda::cudaMemcpy(
                a_,
                a.as_ptr() as *const c_void,
                12,
                cuda::MemcpyKind::HostToDevice
            ), cuda::Error::Success);
            assert_eq!(cuda::cudaMemcpy(
                b_,
                b.as_ptr() as *const c_void,
                12,
                cuda::MemcpyKind::HostToDevice
            ), cuda::Error::Success);

            assert_eq!(cublas::cublasCreate_v2(&mut handle), cublas::Status::Success);
            assert_eq!(cublas::cublasHgemm(
                handle,
                cublas::Operation::N,
                cublas::Operation::N,
                3, 3, 2,
                &c_1,
                b_, 3,
                a_, 2,
                &c_0,
                c_, 3
            ), cublas::Status::Success);
            assert_eq!(cublas::cublasDestroy_v2(handle), cublas::Status::Success);

            // check the results
            assert_eq!(cuda::cudaMemcpy(
                c.as_ptr() as *mut c_void,
                c_,
                18,
                cuda::MemcpyKind::DeviceToHost
            ), cuda::Error::Success);

            assert_eq!(c, [
                f16::from(9.0f32), f16::from(12.0f32), f16::from(15.0f32),
                f16::from(19.0f32), f16::from(26.0f32), f16::from(33.0f32),
                f16::from(29.0f32), f16::from(40.0f32), f16::from(51.0f32)
            ])
        }
    }

    #[test]
    fn sconv() {
        let mut handle: cudnn::Handle = ptr::null_mut();
        let mut convolution: cudnn::ConvolutionDescriptor = ptr::null_mut();
        let mut filter: cudnn::FilterDescriptor = ptr::null_mut();
        let mut inout_tensor: cudnn::TensorDescriptor = ptr::null_mut();

        let mut input = ptr::null_mut();
        let mut output = ptr::null_mut();
        let mut weights = ptr::null_mut();

        let mut workspace_size: usize = 0;
        let mut workspace = ptr::null_mut();

        let c_0: f32 = 0.0;
        let c_1: f32 = 1.0;

        unsafe {
            assert!(cudnn::cudnnCreate(&mut handle).is_ok());

            assert!(cudnn::cudnnCreateTensorDescriptor(&mut inout_tensor).is_ok());
            assert!(cudnn::cudnnSetTensor4dDescriptor(
                inout_tensor,
                cudnn::TensorFormat::NCHW,
                cudnn::DataType::Float,
                1, 256, 19, 19
            ).is_ok());

            assert!(cudnn::cudnnCreateConvolutionDescriptor(&mut convolution).is_ok());
            assert!(cudnn::cudnnSetConvolution2dDescriptor(
                convolution,
                1, 1, 1, 1, 1, 1,
                cudnn::ConvolutionMode::CrossCorrelation,
                cudnn::DataType::Float
            ).is_ok());

            assert!(cudnn::cudnnCreateFilterDescriptor(&mut filter).is_ok());
            assert!(cudnn::cudnnSetFilter4dDescriptor(
                filter,
                cudnn::DataType::Float,
                cudnn::TensorFormat::NCHW,
                256, 256, 3, 3
            ).is_ok());

            assert!(cudnn::cudnnGetConvolutionForwardWorkspaceSize(
                handle,
                inout_tensor,
                filter,
                convolution,
                inout_tensor,
                cudnn::ConvolutionFwdAlgo::Winograd,
                &mut workspace_size
            ).is_ok());

            assert!(cuda::cudaMalloc(&mut input, 4 * 92416).is_ok());
            assert!(cuda::cudaMemcpy(
                input,
                vec! [1.0f32; 92416].as_ptr() as *const c_void,
                4 * 92416, cuda::MemcpyKind::HostToDevice
            ).is_ok());
            assert!(cuda::cudaMalloc(&mut weights, 4 * 589824).is_ok());
            assert!(cuda::cudaMemcpy(
                weights,
                vec! [1.0f32; 589824].as_ptr() as *const c_void,
                4 * 589824, cuda::MemcpyKind::HostToDevice
            ).is_ok());

            assert!(cuda::cudaMalloc(&mut output, 4 * 92416).is_ok());
            assert!(cuda::cudaMalloc(&mut workspace, workspace_size).is_ok());

            assert!(cudnn::cudnnConvolutionForward(
                handle,
                &c_1,
                inout_tensor, input,
                filter, weights,
                convolution,
                cudnn::ConvolutionFwdAlgo::Winograd,
                workspace, workspace_size,
                &c_0,
                inout_tensor, output
            ).is_ok());

            let ones = vec! [0.0f32; 92416];

            assert!(cuda::cudaMemcpy(
                ones.as_ptr() as *mut c_void,
                output,
                4 * 92416, cuda::MemcpyKind::DeviceToHost
            ).is_ok());

            for one in ones.iter() {
                assert!(*one == 1024.0
                    || *one == 1536.0
                    || *one == 2304.0,
                    "{}", *one);
            }
        }
    }
}
