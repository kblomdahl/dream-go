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

use crate::Variable;
use super::BuilderParseErr;

use std::convert::TryFrom;
use std::slice;
use std::str::FromStr;

use dg_utils::b85;
use dg_utils::json::{JsonKey, JsonToken};
use dg_utils::types::f16;

#[derive(Debug)]
enum VariableType {
    Half,
    Float,
    Integer
}

impl FromStr for VariableType {
    type Err = BuilderParseErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f2" => Ok(Self::Half),
            "f4" => Ok(Self::Float),
            "i4" => Ok(Self::Integer),
            _ => Err(BuilderParseErr::UnrecognizedVariableType(s.to_string()))
        }
    }
}

#[derive(Debug)]
enum VariableValue {
    Half(Vec<f16>),
    Float(Vec<f32>),
    Integer(Vec<i32>)
}

#[derive(Debug)]
pub struct VariableBuilder {
    variable_type: VariableType,
    value: VariableValue
}

impl Default for VariableBuilder {
    fn default() -> Self {
        Self {
            variable_type: VariableType::Float,
            value: VariableValue::Float(vec! [])
        }
    }
}

impl VariableBuilder {
    pub fn parse(&mut self, stack: &[JsonKey], token: &JsonToken) -> Result<(), BuilderParseErr> {
        match (stack, token) {
            (_, JsonToken::ObjectStart) => {},
            (_, JsonToken::ObjectEnd) => {},
            ([JsonKey::Object(name)], token) if name == "t" => {
                self.variable_type = String::try_from(token).map_err(|_| BuilderParseErr::UnrecognizedFormat).and_then(|s| s.parse::<VariableType>())?;
            },
            ([JsonKey::Object(name)], _) if name == "s" => {
                // pass
            },
            ([JsonKey::Object(name)], JsonToken::StringPtr { ptr, len }) if name == "v" => {
                let s = unsafe { slice::from_raw_parts(*ptr, *len) };

                match self.variable_type {
                    VariableType::Half => {
                        let value = b85::decode::<f16, f16>(s).ok_or(BuilderParseErr::UnrecognizedVariableValue)?;
                        self.value = VariableValue::Half(value);
                    },
                    VariableType::Float => {
                        let value = b85::decode::<f32, f32>(s).ok_or(BuilderParseErr::UnrecognizedVariableValue)?;
                        self.value = VariableValue::Float(value);
                    },
                    VariableType::Integer => {
                        let value = b85::decode::<i32, i32>(s).ok_or(BuilderParseErr::UnrecognizedVariableValue)?;
                        self.value = VariableValue::Integer(value);
                    },
                }
            },

            _ => { return Err(BuilderParseErr::UnrecognizedFormat) }
        }

        Ok(())
    }

    pub fn build(self) -> Variable {
        match self.value {
            VariableValue::Float(v) => Variable::from(v),
            VariableValue::Half(v) => Variable::from(v),
            VariableValue::Integer(v) => Variable::from(v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use dg_utils::json::JsonStream;

    fn parse(example: &[u8]) -> Variable {
        let mut builder = VariableBuilder::default();

        for entry in JsonStream::new(example) {
            builder.parse(&entry.stack()[..], entry.token()).expect("parse error");
        }

        builder.build()
    }

    #[test]
    fn parse_float() {
        let example = b"{\"t\": \"f4\", \"v\": \"000<4\"}";

        assert_eq!(
            parse(example),
            Variable::from(vec! [9.5])
        );
    }

    #[test]
    fn parse_half() {
        let example = b"{\"t\": \"f2\", \"v\": \"gh5$D\"}";

        assert_eq!(
            parse(example),
            Variable::from(vec! [f16::from(2.7578125), f16::from(3.67382812)])
        );
    }

    #[test]
    fn parse_integer() {
        let example = b"{\"t\": \"i4\", \"v\": \"*8l(i\"}";

        assert_eq!(
            parse(example),
            Variable::from(vec! [-42])
        );
    }
}
