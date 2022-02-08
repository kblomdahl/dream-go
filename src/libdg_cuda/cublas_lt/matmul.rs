// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use crate::cublas_lt::*;
use crate::{cudaStream_t, Stream};

use libc::{c_void, size_t};

#[allow(non_camel_case_types)]
pub type cublasLtMatrixLayout_t = *const c_void;

#[link(name = "cublasLt")]
extern {
    fn cublasLtMatmul(
        light_handle: cublasLtHandle_t,
        compute_desc: cublasLtMatmulDesc_t,
        alpha: *const c_void,
        a: *const c_void,
        a_desc: cublasLtMatrixLayout_t,
        b: *const c_void,
        b_desc: cublasLtMatrixLayout_t,
        beta: *const c_void,
        c: *const c_void,
        d_desc: cublasLtMatrixLayout_t,
        d: *mut c_void,
        d_desc: cublasLtMatrixLayout_t,
        algo: *const cublasLtMatmulAlgo_t,
        workspace: *mut c_void,
        workspace_size_in_bytes: size_t,
        stream: cudaStream_t
    ) -> cublasStatus_t;
}

pub struct Matmul {
    compute_desc: MatmulDesc,
    alpha: [f32; 2],
    a_desc: MatrixLayout,
    b_desc: MatrixLayout,
    c_desc: MatrixLayout,
    d_desc: MatrixLayout,
    algo: MatmulAlgo
}

unsafe impl Send for Matmul {}

impl Matmul {
    pub fn new(
        light_handle: &Handle,
        compute_desc: MatmulDesc,
        alpha: [f32; 2],
        a_desc: MatrixLayout,
        b_desc: MatrixLayout,
        c_desc: MatrixLayout,
        d_desc: MatrixLayout,
    ) -> Result<Self, Status>
    {
        let algo = MatmulAlgo::new(
            light_handle,
            &compute_desc,
            &a_desc,
            &b_desc,
            &c_desc,
            &d_desc,
            &MatmulPreference::new()?
        )?;

        Ok(Self {
            compute_desc,
            alpha,
            a_desc,
            b_desc,
            c_desc,
            d_desc,
            algo
        })
    }

    pub fn desc(&self) -> &MatmulDesc {
        &self.compute_desc
    }

    pub fn algo(&self) -> &MatmulAlgo {
        &self.algo
    }

    pub fn forward(
        &self,
        light_handle: &Handle,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        d: *mut c_void,
        workspace: *mut c_void,
        workspace_size_in_bytes: size_t,
        stream: &Stream
    ) -> Result<(), Status>
    {
        let status = unsafe {
            cublasLtMatmul(
                **light_handle,
                *self.compute_desc,
                &self.alpha[0] as *const f32 as *const _,
                a,
                *self.a_desc,
                b,
                *self.b_desc,
                &self.alpha[1] as *const f32  as *const _,
                c,
                *self.c_desc,
                d,
                *self.d_desc,
                &*self.algo as *const _,
                workspace,
                workspace_size_in_bytes,
                **stream
            )
        };

        status.into_result(())
    }
}

#[cfg(test)]
mod tests {
    use crate as cuda;
    use super::*;

    #[test]
    fn check_matmul_4x4() -> Result<(), Status> {
        let light_handle = Handle::new()?;
        let stream = Stream::default();
        let bias = cuda::Ptr::from_slice(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &stream).unwrap();
        let matmul_desc = MatmulDesc::new(ComputeType::Real32F, DataType::Real32F)?
            .with_epilogue(Epilogue::ReluBias)?
            .with_bias(&bias)?
            .with_transpose_a(Operation::Transpose)?
            .with_transpose_b(Operation::Transpose)?;
        let a_desc = MatrixLayout::new(DataType::Real32F, 4, 4, 4)?;
        let b_desc = MatrixLayout::new(DataType::Real32F, 4, 4, 4)?;
        let c_desc = MatrixLayout::new(DataType::Real32F, 4, 4, 4)?;
        let d_desc = MatrixLayout::new(DataType::Real32F, 4, 4, 4)?;
        let matmul = Matmul::new(
            &light_handle,
            matmul_desc,
            [1.0, 0.0],
            a_desc,
            b_desc,
            c_desc,
            d_desc,
        ).unwrap();

        let workspace = cuda::Ptr::new(matmul.algo().workspace_size_in_bytes()).unwrap();
        let a = cuda::Ptr::from_slice(&[
            5.0f32, 7.0f32, 9.0f32, 10.0f32,
            2.0f32, 3.0f32, 3.0f32, 8.0f32,
            8.0f32, 10.0f32, 2.0f32, 3.0f32,
            3.0f32, 3.0f32, 4.0f32, 8.0f32
        ], &stream).unwrap();
        let b = cuda::Ptr::from_slice(&[
            3.0f32, 10.0f32, 12.0f32, 18.0f32,
            12.0f32, 1.0f32, 4.0f32, 9.0f32,
            9.0f32, 10.0f32, 12.0f32, 2.0f32,
            3.0f32, 12.0f32, 4.0f32, 10.0f32
        ], &stream).unwrap();
        let c = cuda::Ptr::from_slice(&[0.0f32; 16], &stream).unwrap();
        let d = cuda::Ptr::from_slice(&[0.0f32; 16], &stream).unwrap();

        matmul.forward(
            &light_handle,
            a.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            d.as_ptr(),
            workspace.as_ptr(),
            workspace.size_in_bytes(),
            &stream
        ).unwrap();

        assert_eq!(
            d.to_vec::<f32>(&stream).unwrap(),
            vec! [
                211.0f32, 95.0f32, 174.0f32, 109.0f32,
                268.0f32, 151.0f32, 149.0f32, 173.0f32,
                237.0f32, 106.0f32, 175.0f32, 132.0f32,
                272.0f32, 151.0f32, 271.0f32, 173.0f32
            ]
        );

        Ok(())
    }
}