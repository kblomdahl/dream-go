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

use crate::{Config, Err, Model};
#[cfg(test)] use crate::Variable;
use super::{BuilderParseErr, TestBuilder, LayersBuilder};

#[cfg(test)] use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::convert::TryFrom;
use std::io::Read;

use dg_utils::json::{JsonKey, JsonToken, JsonStream};

#[derive(Debug)]
pub struct Builder {
    config: Config,
    representation: LayersBuilder,
    dynamics: LayersBuilder,
    gru: LayersBuilder,
    prediction: LayersBuilder,
    test: TestBuilder
}

impl Default for Builder {
    fn default() -> Self {
        let paths = vec! [
            // check for a file named the same as the current executable, but
            // with a `.json` extension.
            env::current_exe().ok().and_then(|file| {
                let mut json = file.clone();
                json.set_extension("json");
                json.as_path().to_str().map(|s| s.to_string())
            }).unwrap_or_else(|| "dream_go.json".to_string()),

            // hard-coded paths to make development and deployment easier
            "dream_go.json".to_string(),
            "models/dream_go.json".to_string(),
            "/usr/share/dreamgo/dream_go.json".to_string(),
            "/usr/share/dream_go/dream_go.json".to_string()
        ];

        paths.into_iter()
            .find_map(|path| {
                if let Ok(f) = File::open(&path) {
                    Some(
                        Builder::parse(f).expect(&format!("could not parse model -- {}", path))
                    )
                } else {
                    None
                }
            })
            .expect("no model found")
    }
}

impl Builder {
    fn empty() -> Self {
        Self {
            config: Config::default(),
            representation: LayersBuilder::new("r"),
            dynamics: LayersBuilder::new("d"),
            gru: LayersBuilder::new("g"),
            prediction: LayersBuilder::new("p"),
            test: TestBuilder::default()
        }
    }

    fn parse_config(mut self, key: &str, token: &JsonToken) -> Result<Self, BuilderParseErr> {
        if key == "embeddings_size" {
            if let Ok(embeddings_size) = usize::try_from(token) {
                self.config.embeddings_size = embeddings_size;
                Ok(self)
            } else {
                Err(BuilderParseErr::UnrecognizedFormat)
            }
        } else if key == "num_features" {
            if let Ok(num_features) = usize::try_from(token) {
                self.config.num_features = num_features;
                Ok(self)
            } else {
                Err(BuilderParseErr::UnrecognizedFormat)
            }
        } else if key == "num_repr_channels" {
            if let Ok(num_repr_channels) = usize::try_from(token) {
                self.config.num_repr_channels = num_repr_channels;
                Ok(self)
            } else {
                Err(BuilderParseErr::UnrecognizedFormat)
            }
        } else if key == "num_dyn_channels" {
            if let Ok(num_dyn_channels) = usize::try_from(token) {
                self.config.num_dyn_channels = num_dyn_channels;
                Ok(self)
            } else {
                Err(BuilderParseErr::UnrecognizedFormat)
            }
        } else {
            Err(BuilderParseErr::UnrecognizedConfig(key.to_string()))
        }
    }

    pub fn parse<R: Read>(reader: R) -> Result<Self, Err> {
        let mut out = Self::empty();

        for entry in JsonStream::new(reader) {
            match (&entry.stack()[..], entry.token()) {
                (_, JsonToken::ArrayStart) => {},
                (_, JsonToken::ArrayEnd) => {},
                (_, JsonToken::ObjectStart) => {},
                (_, JsonToken::ObjectEnd) => {},

                ([JsonKey::Object(name), JsonKey::Object(key)], token) if name == "c" => {
                    out = out.parse_config(key, token)?;
                },

                ([JsonKey::Object(name), JsonKey::Object(network), stack @ ..], token) if name == "n" => {
                    match network.as_str() {
                        "r" => { out.representation.parse(stack, token)? },
                        "d" => { out.dynamics.parse(stack, token)? },
                        "g" => { out.gru.parse(stack, token)? },
                        "p" => { out.prediction.parse(stack, token)? },
                        _ => { return Err(BuilderParseErr::UnrecognizedScope(network.to_string()))? }
                    };
                },

                ([JsonKey::Object(name), stack @ ..], token) if name == "t" => {
                    out.test.parse(stack, token)?;
                },

                _ => {
                    return Err(BuilderParseErr::UnrecognizedFormat)?;
                }
            }
        }

        Ok(out)
    }

    pub fn build(self) -> Result<Model, Err> {
        let config = self.config.clone();
        let representation = self.representation.build();
        let dynamics = self.dynamics.build();
        let gru = self.gru.build();
        let prediction = self.prediction.build();

        Model::new(config, representation, dynamics, gru, prediction)
    }

    #[cfg(test)]
    pub fn build_with_tests(self) -> Result<(Model, HashMap<String, Variable>), Err> {
        let config = self.config.clone();
        let representation = self.representation.build();
        let dynamics = self.dynamics.build();
        let gru = self.gru.build();
        let prediction = self.prediction.build();
        let model = Model::new(config, representation, dynamics, gru, prediction)?;
        let test = self.test.build();

        Ok((model, test))
    }
}
