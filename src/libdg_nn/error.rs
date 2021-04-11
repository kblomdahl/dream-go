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

use dg_cuda as cuda;
use dg_cuda::cudnn;

#[derive(Debug)]
pub enum Error {
    CuDNN(cudnn::Status),
    Cuda(cuda::Error),
    MalformedWeights,
    MissingWeights
}

impl From<cuda::Error> for Error {
    fn from(s: cuda::Error) -> Error {
        match s {
            other => Error::Cuda(other)
        }
    }
}

impl From<cudnn::Status> for Error {
    fn from(s: cudnn::Status) -> Error {
        match s {
            other => Error::CuDNN(other)
        }
    }
}
