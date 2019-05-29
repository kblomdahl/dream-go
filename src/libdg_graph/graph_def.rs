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
    Scale
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

#[derive(Clone, Deserialize, Debug, PartialEq)]
pub struct VariableDef {
    pub id: usize,
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
pub struct ConstantDef {
    pub shape: Vec<isize>,
    pub value: ConstantValueDef
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantValueDef {
    pub inner: Arc<Vec<f16>>
}

impl Deref for ConstantValueDef {
    type Target = Vec<f16>;

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
        match b85::decode::<f16, f16>(v.as_bytes()) {
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
      "shape": [-1, 19, 19, 40]
    }
  },
  "output": {
    "policy": {
      "id": 2,
      "shape": [-1, 362]
    },
    "value": {
      "id": 4,
      "shape": [-1, 2]
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
            shape: vec! [-1, 19, 19, 40]
        });

        assert_eq!(g.output.len(), 2);
        assert!(g.output.contains_key("policy"));
        assert_eq!(g.output["policy"], VariableDef {
            id: 2,
            shape: vec! [-1, 362]
        });
        assert!(g.output.contains_key("value"));
        assert_eq!(g.output["value"], VariableDef {
            id: 4,
            shape: vec! [-1, 2]
        });

        assert!(g.layers.is_empty());
    }

    #[test]
    fn test_softmax() {
        let content = r#"
{
  "type": "Softmax",
  "in": [{"id": 66, "shape": [-1, 2]}],
  "output": [{"id": 4, "shape": [-1, 2]}]
}
"#;
        let l: LayerDef = serde_json::from_reader(Cursor::new(content)).unwrap();

        assert_eq!(l.type_of, LayerTypeDef::Softmax);
        assert_eq!(l.input.len(), 1);
        assert_eq!(l.input[0], VariableDef { id: 66, shape: vec![-1, 2]});
        assert_eq!(l.output.len(), 1);
        assert_eq!(l.output[0], VariableDef { id: 4, shape: vec![-1, 2]});
        assert_eq!(l.arguments, None);
    }

    #[test]
    fn test_conv2d() {
        let content = r#"
{
  "type": "Conv2D",
  "in": [{"id": 60, "shape": [-1, 19, 19, 64]}],
  "output": [{"id": 65, "shape": [-1, 19, 19, 2]}],
  "arguments": {
    "bias": {
      "shape": [2],
      "value": "yREt{"
    },
    "kernel": {
      "shape": [2, 1, 1, 64],
      "value": "P_rYjFsKkO?54M$FRNoF<E`ABA1+F+e=voVAG1*|e5eK?o-F{eX)W%iJ*fO7qp6y#1uYt=&#Ja8&8aac13!qYH@=^zGqHcKu&vRoDlulQ1gs!06(^u7Vx)#LZ6!~s-mDv}*RG(h@UUc<VyWD!=A?Tg7pj#k^Rc+3p|TCIB&r%PeW}i;RISLMWU7@Y5-$s^F)@^&1G2&?NvYx{3N6gBf-5|vE~gBo@v8l;JS#<~sjBxZW2VO`biQG%_&x`wJF&X2@vbqfZ!kZuo2*zae<;@}j;QM}WhJSo7pxztXRh0<e6U>&!K7-c$f#8vy{xw?j<KDc"
    }
  }
}
"#;
        let l: LayerDef = serde_json::from_reader(Cursor::new(content)).unwrap();
        let bias_value = vec! [-0.089538574, 0.08947754]
            .into_iter()
            .map(|x| f16::from(x))
            .collect();
        let kernel_value = vec![
            -0.22851563, -0.12927246, -0.032714844, 0.11035156, -0.027038574, -0.007534027,
            -0.056121826, 0.02494812, -0.09197998, -0.0047454834, 0.09564209, -0.0982666,
             0.14050293, -0.0011034012, -0.22253418, 0.09869385, -0.03503418, 0.009811401,
             0.087768555, -0.15625, 0.08453369, -0.027069092, -0.033111572, 0.019470215,
            -0.044036865, -0.071899414, 0.078430176, -0.032043457, -0.05319214, 0.0736084,
            -0.045318604, 0.03274536, 1.7529297, -0.07080078, -1.5537109, -0.029769897,
            -0.16247559, -0.117126465, -0.08886719, -0.07525635, 0.16137695, -0.084350586,
            -0.06274414, 0.111328125, 0.027664185, 0.051757813, -0.017120361, 0.20385742,
             0.021194458, -0.04147339, -0.076049805, -0.06414795, -0.10687256, -0.103515625,
            -0.15429688, -0.0021438599, -0.042053223, -0.053588867, -0.019134521, 0.0146102905,
            -0.047576904, 0.07159424, -0.18591309, -0.018432617, -0.20715332, -0.12658691,
            -0.047973633, 0.12817383, -0.042877197, -0.03048706, -0.08325195, -0.009338379,
            -0.049926758, 0.03579712, 0.11047363, -0.06317139, 0.16223145, -0.00894165,
            -0.18786621, 0.04498291, -0.041290283, 0.026885986, 0.07873535, -0.18115234,
             0.058654785, -0.016540527, -0.028045654, -0.019714355, -0.054229736, -0.093566895,
             0.056518555, -0.028396606, -0.05203247, 0.07757568, -0.02494812, 0.045135498,
            -1.6132813, -0.06842041, 1.7421875, -0.023544312, -0.16345215, -0.12072754,
            -0.10845947, -0.06549072, 0.1385498, -0.097595215, -0.07196045, 0.099121094,
             0.035125732, 0.04562378, -0.035583496, 0.15368652, 0.021072388, -0.03640747,
            -0.06390381, -0.040008545, -0.10003662, -0.0758667, -0.14013672, 0.0003273487,
            -0.018569946, -0.050109863, -0.037353516, 0.005207062, -0.074035645, 0.052459717,
            -0.17358398, -0.00548172
        ]
            .into_iter()
            .map(|x| f16::from(x))
            .collect();

        assert_eq!(l, LayerDef {
            type_of: LayerTypeDef::Conv2D,
            input: vec! [VariableDef { id: 60, shape: vec![-1, 19, 19, 64]}],
            output: vec! [VariableDef { id: 65, shape: vec![-1, 19, 19, 2]}],
            arguments: Some(LayerArgumentsDef {
                kernel: Some(ConstantDef {
                    shape: vec! [2, 1, 1, 64],
                    value: ConstantValueDef { inner: Arc::new(kernel_value) }
                }),
                bias: Some(ConstantDef {
                    shape: vec! [2],
                    value: ConstantValueDef { inner: Arc::new(bias_value) }
                }),
                group_count: 1,
                activation: ActivationTypeDef::Linear,
                alpha: None
            })
        });
    }
}