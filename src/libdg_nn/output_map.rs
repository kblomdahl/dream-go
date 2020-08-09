// Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

pub struct OutputMap {
    value: Vec<f32>,
    policy: Vec<f32>,
}

impl OutputMap {
    pub fn new<T: Sized>(
        value: Vec<T>,
        policy: Vec<T>
    ) -> Self
        where f32: From<T>
    {
        Self {
            value: value.into_iter().map(|x| f32::from(x)).collect(),
            policy: policy.into_iter().map(|x| f32::from(x)).collect()
        }
    }

    pub fn value(&self) -> &Vec<f32> {
        &self.value
    }

    pub fn policy(&self) -> &Vec<f32> {
        &self.policy
    }
}
