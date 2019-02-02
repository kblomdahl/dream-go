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

use super::ffi::{cudnn, cuda, cublas};

#[derive(Debug)]
pub enum Error {
    CuDNN(cudnn::Status),
    Cuda(cuda::Error),
    CuBLAS(cublas::Status),
    MissingWeights
}

impl From<cublas::Status> for Error {
    fn from(s: cublas::Status) -> Error {
        match s {
            cublas::Status::Success => unreachable!(),
            other => Error::CuBLAS(other)
        }
    }
}

impl From<cuda::Error> for Error {
    fn from(s: cuda::Error) -> Error {
        match s {
            cuda::Error::Success => unreachable!(),
            other => Error::Cuda(other)
        }
    }
}

impl From<cudnn::Status> for Error {
    fn from(s: cudnn::Status) -> Error {
        match s {
            cudnn::Status::Success => unreachable!(),
            other => Error::CuDNN(other)
        }
    }
}
