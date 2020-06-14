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

use dg_cuda as cuda2;
use dg_cuda::cudnn as cudnn2;
use super::ffi::cuda;

#[derive(Debug)]
pub enum Error {
    CuDNN2(cudnn2::Status),
    Cuda(cuda::Error),
    Cuda2(cuda2::Error),
    MissingWeights
}

impl From<cuda::Error> for Error {
    fn from(s: cuda::Error) -> Error {
        match s {
            cuda::Error::Success => unreachable!(),
            other => Error::Cuda(other)
        }
    }
}

impl From<cuda2::Error> for Error {
    fn from(s: cuda2::Error) -> Error {
        match s {
            other => Error::Cuda2(other)
        }
    }
}

impl From<cudnn2::Status> for Error {
    fn from(s: cudnn2::Status) -> Error {
        match s {
            other => Error::CuDNN2(other)
        }
    }
}
