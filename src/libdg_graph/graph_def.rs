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

use std::collections::HashMap;
use std::fmt;
use std::io::Read;
use std::iter::{Chain, Zip};
use std::ops::Deref;
use std::slice::Iter;
use std::sync::Arc;

use serde::{Deserialize, Deserializer};
use serde::de::Visitor;
use serde_json;

use dg_utils::b85;
use dg_utils::types::f16;
use std::mem::size_of;

#[derive(Deserialize, Debug)]
pub struct GraphDef {
    #[serde(alias = "inputs")]
    pub input: HashMap<String, VariableDef>,

    #[serde(alias = "outputs")]
    pub output: HashMap<String, VariableDef>,

    #[serde(default = "Vec::new")]
    pub layers: Vec<LayerDef>
}

#[derive(Clone, Copy, Deserialize, Debug, PartialEq)]
pub enum LayerTypeDef {
    Activation,
    Add,
    Conv2D,
    Dense,
    GlobalAveragePooling,
    Identity,
    Multiply,
    Softmax,
    Scale,
    Transform
}

#[derive(Clone, Copy, Deserialize, Debug, PartialEq)]
pub enum ActivationTypeDef {
    ReLU,
    Sigmoid,
    Linear
}

impl ActivationTypeDef {
    fn linear() -> Self {
        ActivationTypeDef::Linear
    }

    fn deserialize<'de, D>(de: D) -> Result<ActivationTypeDef, D::Error>
        where D: Deserializer<'de>
    {
        de.deserialize_str(ActivationTypeDefVisitor)
    }
}

#[derive(Clone, Deserialize, Debug, PartialEq)]
pub struct LayerDef {
    #[serde(alias = "type")]
    pub type_of: LayerTypeDef,

    #[serde(alias = "in")]
    pub input: Vec<VariableDef>,

    #[serde(alias = "out")]
    pub output: Vec<VariableDef>,

    #[serde(default)]
    pub arguments: Option<LayerArgumentsDef>
}

impl LayerDef {
    pub fn variables(&self) -> Chain<Iter<VariableDef>, Iter<VariableDef>> {
        self.input.iter().chain(self.output.iter())
    }

    pub fn map(&self) -> Zip<Iter<VariableDef>, Iter<VariableDef>> {
        self.input.iter().zip(self.output.iter())
    }

    pub fn is_input_half(&self) -> bool {
        self.input.iter().all(|i| i.data_type == DataTypeDef::Half)
    }
}

#[derive(Clone, Deserialize, Debug, PartialEq)]
pub struct LayerArgumentsDef {
    #[serde(default)]
    pub kernel: Option<ConstantDef>,

    #[serde(default)]
    pub bias: Option<ConstantDef>,

    #[serde(default)]
    pub alpha: Option<ConstantDef>,

    #[serde(default = "default_group_count")]
    pub group_count: usize,

    #[serde(default = "ActivationTypeDef::linear", deserialize_with = "ActivationTypeDef::deserialize")]
    pub activation: ActivationTypeDef
}

pub fn default_group_count() -> usize {
    1
}

#[derive(Clone, Copy, Deserialize, Debug, Eq, PartialEq, Hash)]
pub enum DataTypeDef  {
    Float,
    Half
}

impl DataTypeDef {
    pub fn size_of(self) -> usize {
        match self {
            DataTypeDef::Float => size_of::<f32>(),
            DataTypeDef::Half => size_of::<f16>(),
        }
    }
}

fn default_data_type() -> DataTypeDef {
    DataTypeDef::Float
}

#[derive(Clone, Deserialize, Debug, PartialEq)]
pub struct VariableDef {
    pub id: usize,

    #[serde(default = "default_data_type")]
    pub data_type: DataTypeDef,
    pub shape: Vec<isize>
}

impl VariableDef {
    pub fn size(&self) -> usize {
        self.shape.iter()
            .map(|&x| if x < 0 { 1 } else { x as usize })
            .product()
    }
}

#[derive(Clone, Deserialize, Debug, PartialEq)]
#[serde(from = "ValidateConstantDef")]
pub struct ConstantDef {
    pub shape: Vec<isize>,
    pub value: ConstantValueDef,
}

#[derive(Deserialize)]
struct ValidateConstantDef {
    shape: Vec<isize>,
    value: ConstantValueDef,

    #[serde(default)]
    mean: Option<f32>,

    #[serde(default)]
    std: Option<f32>
}

impl From<ValidateConstantDef> for ConstantDef {
    fn from(other: ValidateConstantDef) -> Self {
        let out = ConstantDef {
            shape: other.shape.clone(),
            value: other.value.clone()
        };

        // verify the `mean`
        if let Some(other_mean) = other.mean {
            let num_elements: isize = other.shape.iter()
                .product();
            let total_sum: f32 = other.value.inner.iter()
                .map(|&x| f32::from(x))
                .sum();
            let average = total_sum / num_elements as f32;

            assert!(other_mean > average - 1e-2, "Actual: {}, Expected: {}", average, other_mean);
            assert!(other_mean < average + 1e-2, "Actual: {}, Expected: {}", average, other_mean);

            // verify the `std`
            if let Some(other_std) = other.std {
                let total_diff: f32 = other.value.inner.iter()
                    .map(|&x| (f32::from(x) - average).abs())
                    .map(|x| x * x)
                    .sum();
                let std = (total_diff / num_elements as f32).sqrt();

                assert!(other_std > std - 1e-2, "Actual: {}, Expected: {}", std, other_std);
                assert!(other_std < std + 1e-2, "Actual: {}, Expected: {}", std, other_std);
            }
        }

        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantValueDef {
    pub inner: Arc<Vec<f32>>
}

impl Deref for ConstantValueDef {
    type Target = Vec<f32>;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

struct ConstantValueDefVisitor;

impl<'de> Visitor<'de> for ConstantValueDefVisitor {
    type Value = ConstantValueDef;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a ConstantValueDef")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where E: serde::de::Error,
    {
        match b85::decode::<f32, f32>(v.as_bytes()) {
            None => Err(serde::de::Error::invalid_value(serde::de::Unexpected::Str(v), &self)),
            Some(vec) => Ok(ConstantValueDef { inner: Arc::new(vec) })
        }
    }
}

impl<'de> Deserialize<'de> for ConstantValueDef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>,
    {
        deserializer.deserialize_str(ConstantValueDefVisitor)
    }
}

struct ActivationTypeDefVisitor;

impl<'de> Visitor<'de> for ActivationTypeDefVisitor {
    type Value = ActivationTypeDef;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an ActivationTypeDef")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where E: serde::de::Error,
    {
        let str = v.to_lowercase();

        if str == "relu" {
            Ok(ActivationTypeDef::ReLU)
        } else if str == "sigmoid" {
            Ok(ActivationTypeDef::Sigmoid)
        } else if str == "linear" {
            Ok(ActivationTypeDef::Linear)
        } else {
            Err(serde::de::Error::invalid_value(serde::de::Unexpected::Str(v), &self))
        }
    }
}

impl GraphDef {
    pub fn from_reader<R: Read>(rd: R) -> Result<GraphDef, serde_json::Error> {
        serde_json::from_reader(rd)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn test_headers() {
        let content = r#"
{
  "input": {
    "features": {
      "id": 0,
      "shape": [-1, 19, 19, 40],
      "data_type": "Float"
    }
  },
  "output": {
    "policy": {
      "id": 2,
      "shape": [-1, 362],
      "data_type": "Float"
    },
    "value": {
      "id": 4,
      "shape": [-1, 2],
      "data_type": "Float"
    }
  },
  "layers": []
}
"#;
        let g = GraphDef::from_reader(Cursor::new(content)).unwrap();

        assert_eq!(g.input.len(), 1);
        assert!(g.input.contains_key("features"));
        assert_eq!(g.input["features"], VariableDef {
            id: 0,
            shape: vec! [-1, 19, 19, 40],
            data_type: DataTypeDef::Float
        });

        assert_eq!(g.output.len(), 2);
        assert!(g.output.contains_key("policy"));
        assert_eq!(g.output["policy"], VariableDef {
            id: 2,
            shape: vec! [-1, 362],
            data_type: DataTypeDef::Float
        });
        assert!(g.output.contains_key("value"));
        assert_eq!(g.output["value"], VariableDef {
            id: 4,
            shape: vec! [-1, 2],
            data_type: DataTypeDef::Float
        });

        assert!(g.layers.is_empty());
    }

    #[test]
    fn test_softmax() {
        let content = r#"
{
  "type": "Softmax",
  "in": [{"id": 66, "shape": [-1, 2], "data_type": "Half"}],
  "output": [{"id": 4, "shape": [-1, 2], "data_type": "Half"}]
}
"#;
        let l: LayerDef = serde_json::from_reader(Cursor::new(content)).unwrap();

        assert_eq!(l.type_of, LayerTypeDef::Softmax);
        assert_eq!(l.input.len(), 1);
        assert_eq!(l.input[0], VariableDef { id: 66, shape: vec![-1, 2], data_type: DataTypeDef::Half});
        assert_eq!(l.output.len(), 1);
        assert_eq!(l.output[0], VariableDef { id: 4, shape: vec![-1, 2], data_type: DataTypeDef::Half});
        assert_eq!(l.arguments, None);
    }
}