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

use crate::BuilderParseErr;

use dg_cuda::{self as cuda, cudnn, cublas_lt};

#[derive(Debug)]
pub enum Err {
    Cuda(cuda::Error),
    Cudnn(cudnn::Status),
    CublasLt(cublas_lt::Status),
    Parse(BuilderParseErr),
    MissingVariable(String),
    UnexpectedValue
}

impl From<cuda::Error> for Err {
    fn from(original: cuda::Error) -> Self {
        Self::Cuda(original)
    }
}

impl From<cudnn::Status> for Err {
    fn from(original: cudnn::Status) -> Self {
        Self::Cudnn(original)
    }
}

impl From<cublas_lt::Status> for Err {
    fn from(original: cublas_lt::Status) -> Self {
        Self::CublasLt(original)
    }
}

impl From<BuilderParseErr> for Err {
    fn from(original: BuilderParseErr) -> Self {
        Self::Parse(original)
    }
}