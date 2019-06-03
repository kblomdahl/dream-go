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

mod activation;
mod conv2d;
mod dense;
mod global_avg_pooling;
mod identity;
mod scale;
mod softmax;
mod op_tensor;
#[cfg(test)] mod tests;

pub use self::scale::Scale;
pub use self::softmax::Softmax;
pub use self::op_tensor::OpTensor;
pub use self::identity::Identity;
pub use self::activation::Activation;
pub use self::conv2d::Conv2D;
pub use self::dense::Dense;
pub use self::global_avg_pooling::GlobalAveragePooling;
