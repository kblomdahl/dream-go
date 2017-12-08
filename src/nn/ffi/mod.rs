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
}
