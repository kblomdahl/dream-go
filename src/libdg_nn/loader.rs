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
use std::io::{BufReader, BufRead, ErrorKind};
use std::path::Path;
use memchr::memchr;

use super::tensor::Tensor;
use super::Error;
use dg_utils::types::f16;
use dg_utils::b85;

/// Step the iterator forward until the character given `stop` character is
/// encountered. The character `stop` is also skipped.
///
/// The implementation is a slight adaptation of `BufRead::read_until` to
/// not include `stop`, and to use `memchr` from the external crate for better
/// performance.
/// 
/// # Arguments
/// 
/// * `iter` - the iterator to step forward
/// * `stop` - the character to step until
/// 
fn skip_until<R: BufRead>(buf_read: &mut R, stop: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);

    loop {
        let (done, used) = {
            let available = match buf_read.fill_buf() {
                Ok(n) => n,
                Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => panic!(e)
            };

            match memchr(stop, available) {
                Some(i) => {
                    out.extend_from_slice(&available[..i]);
                    (true, i + 1)
                },
                None => {
                    out.extend_from_slice(available);
                    (false, available.len())
                }
            }
        };

        buf_read.consume(used);
        if done || used == 0 {
            return out;
        }
    }
}

/// An iterator that parse entries with the following format:
/// 
/// `"name": { "s": "...", v: "..." }`
/// 
struct JsonEntryIter<R: BufRead> {
    buf_read: R
}

impl<R: BufRead> Iterator for JsonEntryIter<R> {
    type Item = (String, Result<Tensor, Error>);

    fn next(&mut self) -> Option<Self::Item> {
        // skip until the quote before the name
        skip_until(&mut self.buf_read, b'"');

        let name = String::from_utf8(skip_until(&mut self.buf_read, b'"')).unwrap();
        if name.is_empty() {
            return None;
        }

        // skip until the next `{` and then parse the interior of the
        // object by iterating over the properties
        skip_until(&mut self.buf_read, b'{');

        let mut tensor = Tensor::default();

        loop {
            skip_until(&mut self.buf_read, b'"');
            let key = String::from_utf8(skip_until(&mut self.buf_read, b'"')).unwrap();

            skip_until(&mut self.buf_read, b'"');
            let value = skip_until(&mut self.buf_read, b'"');

            if key == "s" {
                let array = b85::decode::<f32, f32>(&value).unwrap();

                tensor.set_scale(array[0]);
            } else if key == "v" {
                match tensor.set_host(b85::decode::<f16, f16>(&value).unwrap()) {
                    Ok(()) => (),
                    Err(reason) => { return Some((name, Err(reason))) }
                }
            } else {
                break
            }

            // check if the object terminated
            let more = skip_until(&mut self.buf_read, b',');
            if memchr(b'}', &more).is_some() {
                break
            }
        };

        Some((name, Ok(tensor)))
    }
}

/// Load all tensors in the given buffer and returns a map from
/// their name to description. If we failed to load any tensors
/// from the given file then `None` is returned.
/// 
/// # Arguments
/// 
/// * `path` -
/// 
fn load_aux<R: BufRead>(reader: R) -> Result<HashMap<String, Tensor>, Error> {
    let mut out: HashMap<String, Tensor> = HashMap::new();
    let iter = JsonEntryIter { buf_read: reader };

    for (name, t) in iter {
        out.insert(name, t?);
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
        load_aux(BufReader::new(file))
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
