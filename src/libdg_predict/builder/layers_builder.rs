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

use crate::{Err, Layer};
use super::{BuilderParseErr, LayerBuilder};

use dg_utils::json::{JsonKey, JsonToken};

#[derive(Debug, Default)]
pub struct LayersBuilder {
    name: String,
    layers: Vec<LayerBuilder>
}

impl LayersBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            layers: vec! []
        }
    }

    pub fn parse(&mut self, stack: &[JsonKey], token: &JsonToken) -> Result<(), BuilderParseErr> {
        match (stack, token) {
            (_, JsonToken::ArrayStart) => {},
            (_, JsonToken::ArrayEnd) => {},
            ([JsonKey::Array(index), stack @ ..], token) => {
                while self.layers.len() <= *index {
                    self.layers.push(LayerBuilder::default());
                }

                self.layers[*index].with_name(&format!("{}[{}]", self.name, index)).parse(stack, token)?;
            },
            _ => return Err(BuilderParseErr::UnrecognizedFormat)
        }

        Ok(())
    }

    pub fn build(mut self) -> Result<Vec<Layer>, Err> {
        let mut out = vec! [];

        for layer in self.layers.drain(..) {
            out.push(layer.build()?);
        }

        Ok(out)
    }
}
