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

use libc::{size_t, c_int};
use std::ops::Deref;

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct cublasLtMatmulAlgo_t {
    data: [u64; 8]
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct cublasLtMatmulHeuristicResult_t {
    algo: cublasLtMatmulAlgo_t,
    workspace_size: size_t,
    state: cublasStatus_t,
    waves_count: f32,
    reserved: [c_int; 4]
}

#[link(name = "cublasLt")]
extern {
    fn cublasLtMatmulAlgoGetHeuristic(
        light_handle: cublasLtHandle_t,
        operation_desc: cublasLtMatmulDesc_t,
        a_desc: cublasLtMatrixLayout_t,
        b_desc: cublasLtMatrixLayout_t,
        c_desc: cublasLtMatrixLayout_t,
        d_desc: cublasLtMatrixLayout_t,
        preference: cublasLtMatmulPreference_t,
        requested_algo_count: c_int,
        heuristic_results_array: *mut cublasLtMatmulHeuristicResult_t,
        return_algo_count: *mut c_int
    ) -> cublasStatus_t;
  }

pub struct MatmulAlgo {
    algo: cublasLtMatmulAlgo_t,
    workspace_size_in_bytes: usize
}

unsafe impl Send for MatmulAlgo {}

impl MatmulAlgo {
    pub fn new(
        light_handle: &Handle,
        operation_desc: &MatmulDesc,
        a_desc: &MatrixLayout,
        b_desc: &MatrixLayout,
        c_desc: &MatrixLayout,
        d_desc: &MatrixLayout,
        preference: &MatmulPreference,
    ) -> Result<Self, Status>
    {
        let mut out = cublasLtMatmulHeuristicResult_t {
            algo: cublasLtMatmulAlgo_t { data: [0; 8] },
            workspace_size: 0,
            state: cublasStatus_t::success(),
            waves_count: 0.0,
            reserved: [0; 4]
        };
        let mut count = 0;
        let status = unsafe {
            cublasLtMatmulAlgoGetHeuristic(
                **light_handle,
                **operation_desc,
                **a_desc,
                **b_desc,
                **c_desc,
                **d_desc,
                **preference,
                1,
                &mut out,
                &mut count
            )
        };
        assert_eq!(count, 1);

        status.into_result(Self { algo: out.algo, workspace_size_in_bytes: out.workspace_size })
    }

    pub fn workspace_size_in_bytes(&self) -> usize {
        self.workspace_size_in_bytes
    }
}

impl Deref for MatmulAlgo {
    type Target = cublasLtMatmulAlgo_t;

    fn deref(&self) -> &Self::Target {
        &self.algo
    }
}
