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

use crate::layers;
use crate::{Err, Layer};
use super::{BuilderParseErr, VariableBuilder};

use std::collections::HashMap;
use std::convert::TryFrom;
use std::str::FromStr;

use dg_utils::json::{JsonKey, JsonToken};

#[derive(Clone, Debug)]
enum LayerType {
    Empty,
    Conv,
    Dense,
    Prediction,
    ResidualBlock,
    Gru
}

impl Default for LayerType {
    fn default() -> Self {
        Self::Empty
    }
}

impl FromStr for LayerType {
    type Err = BuilderParseErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "conv" => Ok(Self::Conv),
            "dense" => Ok(Self::Dense),
            "pred" => Ok(Self::Prediction),
            "residual_block" => Ok(Self::ResidualBlock),
            "gru" => Ok(Self::Gru),
            _ => Err(BuilderParseErr::UnrecognizedLayerType(s.to_string()))
        }
    }
}

#[derive(Debug, Default)]
pub struct LayerBuilder {
    name: String,
    layer_type: LayerType,
    variables: HashMap<String, VariableBuilder>
}

impl LayerBuilder {
    pub fn with_name(&mut self, name: &str) -> &mut Self {
        self.name = name.to_string();
        self
    }

    pub fn parse(&mut self, stack: &[JsonKey], token: &JsonToken) -> Result<(), BuilderParseErr> {
        match (stack, token) {
            (_, JsonToken::ObjectStart) => {},
            (_, JsonToken::ObjectEnd) => {},
            ([JsonKey::Object(name)], token) if name == "t" => {
                self.layer_type = String::try_from(token).map_err(|_| BuilderParseErr::UnrecognizedFormat).and_then(|s| s.parse::<LayerType>())?;
            },
            ([JsonKey::Object(name), JsonKey::Object(variable_name), stack @ ..], token) if name == "vs" => {
                let variable_name = variable_name.to_string();

                self.variables.entry(variable_name).or_default().parse(stack, token)?;
            },
            _ => { return Err(BuilderParseErr::UnrecognizedFormat) }
        }

        Ok(())
    }

    pub fn build(mut self) -> Result<Layer, Err> {
        let variables = self.variables.drain()
            .map(|(key, var)| (key, var.build()))
            .collect::<HashMap<_, _>>();

        match self.layer_type {
            LayerType::Conv => Layer::new(&self.name, layers::ConvFactory::default(), variables),
            LayerType::Dense => Layer::new(&self.name, layers::DenseFactory::default(), variables),
            LayerType::Prediction => Layer::new(&self.name, layers::PredictionFactory::default(), variables),
            LayerType::ResidualBlock => Layer::new(&self.name, layers::ResidualBlockFactory::default(), variables),
            LayerType::Gru => Layer::new(&self.name, layers::GruFactory::default(), variables),
            LayerType::Empty => unreachable!()
        }
    }
}
