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

#[derive(Clone, Debug)]
pub struct Config {
    pub num_features: usize,
    pub embeddings_size: usize,
    pub num_repr_channels: usize,
    pub num_dyn_channels: usize,
    pub image_size: usize
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_features: 0,
            embeddings_size: 0,
            num_repr_channels: 0,
            num_dyn_channels: 0,
            image_size: 19
        }
    }
}

impl Config {
    pub fn with_num_features(mut self, num_features: usize) -> Self {
        self.num_features = num_features;
        self
    }

    pub fn with_image_size(mut self, image_size: usize) -> Self {
        self.image_size = image_size;
        self
    }

    pub fn with_embeddings_size(mut self, embeddings_size: usize) -> Self {
        self.embeddings_size = embeddings_size;
        self
    }
}