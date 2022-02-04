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

use super::{BuilderParseErr, VariableBuilder};
use crate::Variable;

use std::collections::HashMap;

use dg_utils::json::{JsonKey, JsonToken};

#[derive(Debug, Default)]
pub struct TestBuilder {
    test: HashMap<String, VariableBuilder>
}

impl TestBuilder {
    pub fn parse(&mut self, stack: &[JsonKey], token: &JsonToken) -> Result<(), BuilderParseErr> {
        match (stack, token) {
            (_, JsonToken::ObjectStart) => {},
            (_, JsonToken::ObjectEnd) => {},
            ([JsonKey::Object(variable_name), stack @ ..], token) => {
                let variable_name = variable_name.to_string();

                self.test.entry(variable_name).or_default().parse(stack, token)?;
            },
            _ => { return Err(BuilderParseErr::UnrecognizedFormat) }
        }

        Ok(())
    }

    pub fn build(mut self) -> HashMap<String, Variable> {
        self.test.drain().map(|(k, v)| (k, v.build())).collect()
    }
}
