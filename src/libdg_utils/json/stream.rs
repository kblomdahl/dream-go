// Copyright 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use super::token_stream::*;

use std::io::Read;

#[derive(Clone, Debug, PartialEq)]
pub enum JsonKey {
    Object(String),
    Array(usize)
}

#[derive(Debug, PartialEq)]
pub struct JsonEntry {
    stack: Vec<JsonKey>,
    token: JsonToken
}

impl JsonEntry {
    fn new(stack: Vec<JsonKey>, token: JsonToken) -> Self {
        Self { stack, token }
    }

    pub fn stack(&self) -> &Vec<JsonKey> {
        &self.stack
    }

    pub fn token(&self) -> &JsonToken {
        &self.token
    }

    #[cfg(test)]
    fn to_owned(self) -> (Vec<JsonKey>, JsonToken) {
        (
            self.stack,
            self.token.to_owned()
        )
    }
}

/// JSON reader that parses a sub-set of the standard very very quickly in a
/// without creating the entire document. Instead streaming each attribute as it
/// appears in the JSON structure.
pub struct JsonStream<R: Read> {
    tokenizer: JsonTokenStream<R>,
    next_is_value: bool,
    stack: Vec<JsonKey>
}

impl<R: Read> JsonStream<R> {
    pub fn new(reader: R) -> Self {
        let tokenizer = JsonTokenStream::new(reader);
        let next_is_value = true;
        let stack = vec! [];

        Self { tokenizer, next_is_value, stack }
    }
}

impl<'a, R: Read> Iterator for JsonStream<R> {
    type Item = JsonEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let mut so_far = vec! [];

        while let Some(token) = self.tokenizer.next() {
            match (&so_far[..], token) {
                ([], JsonToken::ObjectStart) => {
                    self.next_is_value = false;
                    return Some(JsonEntry::new(self.stack.clone(), JsonToken::ObjectStart))
                },
                ([], JsonToken::ObjectEnd) => {
                    self.stack.pop();
                    self.next_is_value = false;
                    return Some(JsonEntry::new(self.stack.clone(), JsonToken::ObjectEnd))
                },
                ([], JsonToken::ArrayStart) => {
                    let curr_stack = self.stack.clone();
                    self.stack.push(JsonKey::Array(0));
                    self.next_is_value = true;
                    return Some(JsonEntry::new(curr_stack, JsonToken::ArrayStart))
                },
                ([], JsonToken::ArrayEnd) => {
                    self.stack.pop();
                    if let Some(JsonKey::Object(_)) = self.stack.last() {
                        self.next_is_value = false;
                    } else {
                        self.next_is_value = true;
                    }
                    return Some(JsonEntry::new(self.stack.clone(), JsonToken::ArrayEnd))
                },
                ([], JsonToken::Comma) => {
                    if let Some(JsonKey::Array(n)) = self.stack.last_mut() {
                        self.next_is_value = true;
                        *n += 1;
                    } else {
                        self.next_is_value = false;
                        self.stack.pop();
                    }
                },
                ([JsonToken::String(key)], JsonToken::Colon) => {
                    self.stack.push(JsonKey::Object(key.clone()));
                    self.next_is_value = true;
                    so_far.clear();
                },
                ([], JsonToken::True) if self.next_is_value => {
                    self.next_is_value = false;
                    return Some(JsonEntry::new(self.stack.clone(), JsonToken::True))
                },
                ([], JsonToken::False) if self.next_is_value => {
                    self.next_is_value = false;
                    return Some(JsonEntry::new(self.stack.clone(), JsonToken::False))
                },
                ([], JsonToken::Null) if self.next_is_value => {
                    self.next_is_value = false;
                    return Some(JsonEntry::new(self.stack.clone(), JsonToken::Null))
                },
                ([], JsonToken::StringPtr { ptr, len }) if self.next_is_value => {
                    self.next_is_value = false;
                    return Some(JsonEntry::new(self.stack.clone(), JsonToken::StringPtr { ptr: ptr, len: len }))
                },
                ([], JsonToken::NumberPtr { ptr, len }) if self.next_is_value => {
                    self.next_is_value = false;
                    return Some(JsonEntry::new(self.stack.clone(), JsonToken::NumberPtr { ptr: ptr, len: len }))
                },
                (_, token) => {
                    so_far.push(token.to_owned());
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_simple_object() {
        let raw = b"{ \"a\": 1, \"b\": [1, 2, 3], \"c\": { \"d\": null } }";
        let mut json = JsonStream::new(&raw[..]);

        assert_eq!(json.next().map(|e| e.to_owned()), Some((vec! [], JsonToken::ObjectStart)));
        assert_eq!(json.next().map(|e| e.to_owned()), Some((vec! [JsonKey::Object("a".into())], JsonToken::Number(1.0))));
        assert_eq!(json.next().map(|e| e.to_owned()), Some((vec! [JsonKey::Object("b".into())], JsonToken::ArrayStart)));
        assert_eq!(json.next().map(|e| e.to_owned()), Some((vec! [JsonKey::Object("b".into()), JsonKey::Array(0)], JsonToken::Number(1.0))));
        assert_eq!(json.next().map(|e| e.to_owned()), Some((vec! [JsonKey::Object("b".into()), JsonKey::Array(1)], JsonToken::Number(2.0))));
        assert_eq!(json.next().map(|e| e.to_owned()), Some((vec! [JsonKey::Object("b".into()), JsonKey::Array(2)], JsonToken::Number(3.0))));
        assert_eq!(json.next().map(|e| e.to_owned()), Some((vec! [JsonKey::Object("b".into())], JsonToken::ArrayEnd)));
        assert_eq!(json.next().map(|e| e.to_owned()), Some((vec! [JsonKey::Object("c".into())], JsonToken::ObjectStart)));
        assert_eq!(json.next().map(|e| e.to_owned()), Some((vec! [JsonKey::Object("c".into()), JsonKey::Object("d".into())], JsonToken::Null)));
        assert_eq!(json.next().map(|e| e.to_owned()), Some((vec! [JsonKey::Object("c".into())], JsonToken::ObjectEnd)));
        assert_eq!(json.next().map(|e| e.to_owned()), Some((vec! [], JsonToken::ObjectEnd)));
        assert_eq!(json.next(), None);
    }
}
