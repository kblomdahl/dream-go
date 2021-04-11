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
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::slice;

use super::tensor::Tensor;
use super::Error;
use dg_utils::types::f16;
use dg_utils::json::{JsonKey, JsonToken, JsonStream};
use dg_utils::b85;

/// Load all tensors in the given buffer and returns a map from
/// their name to description. If we failed to load any tensors
/// from the given file then `None` is returned.
///
/// # Arguments
///
/// * `path` -
///
fn load_aux<R: Read>(reader: R) -> Result<HashMap<String, Tensor>, Error> {
    let mut out: HashMap<String, Tensor> = HashMap::new();

    for entry in JsonStream::new(reader) {
        match (&entry.stack()[..], entry.token()) {
            ([], JsonToken::ObjectStart) => {},
            ([], JsonToken::ObjectEnd) => {},
            ([JsonKey::Object(name)], JsonToken::ObjectStart) => {
                out.insert(name.clone(), Tensor::default());
            },
            ([JsonKey::Object(_)], JsonToken::ObjectEnd) => {},
            ([JsonKey::Object(name), JsonKey::Object(attribute)], JsonToken::StringPtr { ptr, len }) => {
                let value = unsafe { slice::from_raw_parts(*ptr, *len) };
                let tensor = out.get_mut(name).expect("could not get tensor");

                if attribute == "s" {
                    if let Some(parsed_value) = b85::decode::<f32, f32>(&value) {
                        tensor.set_scale(parsed_value[0]);
                    } else {
                        return Err(Error::MalformedWeights);
                    }
                } else if attribute == "v" {
                    let array = b85::decode::<f16, f16>(&value).ok_or(Error::MalformedWeights);

                    if let Err(reason) = array.and_then(|h| tensor.set_host(h)) {
                        return Err(reason);
                    }
                } else {
                    return Err(Error::MalformedWeights);
                }
            }
            _ => { return Err(Error::MalformedWeights) }
        }
    }

    // an empty result-set is an error
    if out.is_empty() {
        Err(Error::MissingWeights)
    } else {
        Ok(out)
    }
}

/// Load all tensors in the given file and returns a map from
/// their name to description. If we failed to load any tensors
/// from the given file then `None` is returned.
///
/// # Arguments
///
/// * `path` -
///
pub fn load(path: &Path) -> Result<HashMap<String, Tensor>, Error> {
    if let Ok(file) = File::open(path) {
        load_aux(file)
    } else {
        Err(Error::MissingWeights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn empty_json() {
        let out = load_aux(Cursor::new(""));

        assert!(out.is_err());
    }

    #[test]
    fn load_json() {
        let out = load_aux(Cursor::new("{\"11v_value/linear_2/offset:0\": {\"s\": \"(^d>V\", \"v\": \"(^d>V\"}}"));
        assert!(out.is_ok());

        // verify internal values
        let out = out.unwrap();

        assert_eq!(out.len(), 1, "{:?}", out.keys().map(|x| x.clone()).collect::<Vec<String>>());
        assert_eq!(out["11v_value/linear_2/offset:0"].scale(), 0.13704996);
        assert_eq!(out["11v_value/linear_2/offset:0"].size_in_bytes(), 4);
    }
}
