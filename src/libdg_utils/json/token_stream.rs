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

use std::convert::TryFrom;
use std::io::{self, Read, ErrorKind};
use std::slice;
use memchr::memchr_iter;

/// The number of bytes to read from the input stream at a time.
const BUF_SIZE: usize = 8 * 1024;

#[derive(Debug, PartialEq)]
pub enum JsonToken {
    StringPtr { ptr: *const u8, len: usize },
    NumberPtr { ptr: *const u8, len: usize },
    String(String),
    Number(f64),
    True,
    False,
    Null,
    Colon,
    Comma,
    ObjectStart,
    ObjectEnd,
    ArrayStart,
    ArrayEnd
}

pub enum JsonTokenErr {
    Unrecognized,
    NotString,
    NotNumber,
}

impl JsonToken {
    pub fn to_owned(self) -> Self {
        match self {
            JsonToken::StringPtr { ptr, len } => {
                let s = unsafe { slice::from_raw_parts(ptr, len) };
                let s = String::from_utf8_lossy(s);

                JsonToken::String(s.to_string())
            },
            JsonToken::NumberPtr { ptr, len } => {
                let s = unsafe { slice::from_raw_parts(ptr, len) };
                let s = String::from_utf8_lossy(s);

                JsonToken::Number(s.parse().expect("could not parse number literal"))
            },
            x => x
        }
    }
}

impl TryFrom<&'_ JsonToken> for String {
    type Error = JsonTokenErr;

    fn try_from(token: &JsonToken) -> Result<Self, Self::Error> {
        match token {
            JsonToken::StringPtr { ptr, len } => {
                let s = unsafe { slice::from_raw_parts(*ptr, *len) };
                let s = String::from_utf8_lossy(s);
                Ok(s.to_string())
            },
            JsonToken::String(s) => Ok(s.clone()),
            _ => Err(JsonTokenErr::NotString)
        }
    }
}

impl TryFrom<&'_ JsonToken> for f64 {
    type Error = JsonTokenErr;

    fn try_from(token: &JsonToken) -> Result<Self, Self::Error> {
        match token {
            JsonToken::NumberPtr { ptr, len } => {
                let s = unsafe { slice::from_raw_parts(*ptr, *len) };
                let s = String::from_utf8_lossy(s);

                s.parse().map_err(|_| JsonTokenErr::Unrecognized)
            },
            JsonToken::Number(f) => Ok(f.clone()),
            _ => Err(JsonTokenErr::NotNumber)
        }
    }
}

impl TryFrom<&'_ JsonToken> for usize {
    type Error = JsonTokenErr;

    fn try_from(token: &JsonToken) -> Result<Self, Self::Error> {
        match token {
            JsonToken::NumberPtr { ptr, len } => {
                let s = unsafe { slice::from_raw_parts(*ptr, *len) };
                let s = String::from_utf8_lossy(s);

                s.parse().map_err(|_| JsonTokenErr::Unrecognized)
            },
            JsonToken::Number(f) => Ok(*f as usize),
            _ => Err(JsonTokenErr::NotNumber)
        }
    }
}

struct JsonTokenReader<R: Read> {
    buf: Vec<u8>,
    reader: R,
}

impl<R: Read> JsonTokenReader<R> {
    fn new(reader: R) -> Self {
        let buf = Vec::with_capacity(2 * BUF_SIZE);

        Self { buf, reader }
    }

    fn consume(&mut self, n: usize) {
        self.buf.drain(..n);
    }

    fn fill_buf(&mut self, force: bool) -> io::Result<&[u8]> {
        if force || self.buf.is_empty() {
            let prev_len = self.buf.len();
            self.buf.resize(prev_len + BUF_SIZE, 0);

            match self.reader.read(&mut self.buf[prev_len..]) {
                Ok(0) => { self.buf.truncate(prev_len) },
                Ok(n) => { self.buf.truncate(prev_len + n) }
                Err(e) => { return Err(e) }
            }
        }

        Ok(&self.buf)
    }
}

pub struct JsonTokenStream<R: Read> {
    start_at: usize,
    buf_reader: JsonTokenReader<R>,
}

impl<R: Read> JsonTokenStream<R> {
    pub fn new(reader: R) -> Self {
        let start_at = 0;
        let buf_reader = JsonTokenReader::new(reader);

        Self { start_at, buf_reader }
    }
}

impl<R: Read> Iterator for JsonTokenStream<R> {
    type Item = JsonToken;

    fn next(&mut self) -> Option<Self::Item> {
        // skip the last token, we need to do this now instead of the previous
        // iteration since this will invalidate the returned references in the
        // token.
        self.buf_reader.consume(self.start_at);

        //
        let mut prev_buf_len = 0;
        let mut reserve_token = None;

        loop {
            let buf = match self.buf_reader.fill_buf(prev_buf_len > 0) {
                Ok(buf) => buf,
                Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(_) => { return None }
            };

            if buf.len() == prev_buf_len {
                self.start_at = buf.len();
                return reserve_token;
            }

            // scan for token in buf
            if let Some(index) = skip_ws(buf) {
                match buf[index] {
                    b'{' => {
                        self.start_at = index + 1;
                        return Some(JsonToken::ObjectStart)
                    },
                    b'}' => {
                        self.start_at = index + 1;
                        return Some(JsonToken::ObjectEnd)
                    },
                    b'[' => {
                        self.start_at = index + 1;
                        return Some(JsonToken::ArrayStart)
                    },
                    b']' => {
                        self.start_at = index + 1;
                        return Some(JsonToken::ArrayEnd)
                    },
                    b':' => {
                        self.start_at = index + 1;
                        return Some(JsonToken::Colon)
                    },
                    b',' => {
                        self.start_at = index + 1;
                        return Some(JsonToken::Comma)
                    },
                    b'"' => {
                        let start_search_at = prev_buf_len.saturating_sub(index + 1);

                        if let Some(i) = parse_string(&buf[(index + 1)..], start_search_at) {
                            self.start_at = index + i + 2;
                            return Some(JsonToken::StringPtr { ptr: &buf[index+1], len: i })
                        } else {
                            // unterminated string, read more data
                        }
                    },
                    b'-' | b'0' | b'1' | b'2' | b'3' | b'4' | b'5' | b'6' | b'7' | b'8' | b'9' => {
                        if let Some(i) = parse_number(&buf[index..]) {
                            self.start_at = index + i + 1;
                            return Some(JsonToken::NumberPtr { ptr: &buf[index], len: i + 1 })
                        } else {
                            // there is no way to know if we've encountered EOF
                            // or just not read the entire token, so if we hit
                            // EOF after this call return this number.
                            reserve_token = Some(JsonToken::NumberPtr { ptr: &buf[index], len: buf.len() - index });
                        }
                    },
                    b't' => {
                        if let Some(i) = parse_literal(&buf[index..], b"true") {
                            self.start_at = index + i;
                            return Some(JsonToken::True)
                        } else {
                            // unterminated number, read more data
                        }
                    },
                    b'f' => {
                        if let Some(i) = parse_literal(&buf[index..], b"false") {
                            self.start_at = index + i;
                            return Some(JsonToken::False)
                        } else {
                            // unterminated number, read more data
                        }
                    },
                    b'n' => {
                        if let Some(i) = parse_literal(&buf[index..], b"null") {
                            self.start_at = index + i;
                            return Some(JsonToken::Null)
                        } else {
                            // unterminated number, read more data
                        }
                    }
                    _ => {
                        panic!("unexpected token -- {:?}", String::from_utf8(buf[index..].to_vec()).unwrap());
                    }
                }
            }

            prev_buf_len = buf.len();
        }
    }
}

/// Returns the index of the next `"` that terminates the string (i.e. is not
/// escaped). If no `"` can be found it returned `None`.
///
/// # Arguments
///
/// * `buf` -
/// * `already_seen` -
///
fn parse_string(buf: &[u8], already_seen: usize) -> Option<usize> {
    for index in memchr_iter(b'"', &buf[already_seen..]) {
        let actual_index = already_seen + index;

        if actual_index == 0 || buf[actual_index-1] != b'\\' {
            // check for the pattern `\\"`
            if actual_index <= 1 || buf[actual_index-2] != b'\\' {
                return Some(actual_index)
            }
        }
    }

    None
}

/// Returns the index of the last character in the number in `buf`.
///
/// # Arguments
///
/// * `buf` -
///
fn parse_number(buf: &[u8]) -> Option<usize> {
    for (i, byte) in buf.iter().enumerate() {
        match byte {
            b'.' => {},
            b'-' | b'+' => {},
            b'e' | b'E' => {},
            b'0' | b'1' | b'2' | b'3' | b'4' | b'5' | b'6' | b'7' | b'8' | b'9' => {},
            _ => { return Some(i - 1) }
        }
    }

    None
}

/// Returns the index of the first characters in `buf` that occurs after
/// `literal`. If there is a mis-match then return `None`.
///
/// # Arguments
///
/// * `buf` -
///
fn parse_literal(buf: &[u8], literal: &[u8]) -> Option<usize> {
    if buf.len() < literal.len() {
        None
    } else {
        for (i, byte) in literal.iter().enumerate() {
            if buf[i] != *byte {
                return None;
            }
        }

        Some(literal.len())
    }
}

/// Return the index of the first byte in `buf` that does not correspond to a
/// whitespace character according to the JSON specification. If the entire
/// buffer consists of only whitespace characters it returns `None`.
///
/// # Arguments
///
/// * `buf` -
///
fn skip_ws(buf: &[u8]) -> Option<usize> {
    for (i, byte) in buf.iter().enumerate() {
        match *byte {
            b' ' | b'\n' | b'\r' | b'\t'  => {},
            _ => { return Some(i) }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_true() {
        let raw = b"true";
        let mut json = JsonTokenStream::new(&raw[..]);

        assert_eq!(json.next(), Some(JsonToken::True));
        assert_eq!(json.next(), None);
    }

    #[test]
    fn stream_false() {
        let raw = b"false";
        let mut json = JsonTokenStream::new(&raw[..]);

        assert_eq!(json.next(), Some(JsonToken::False));
        assert_eq!(json.next(), None);
    }

    #[test]
    fn stream_null() {
        let raw = b"null";
        let mut json = JsonTokenStream::new(&raw[..]);

        assert_eq!(json.next(), Some(JsonToken::Null));
        assert_eq!(json.next(), None);
    }

    #[test]
    fn stream_string() {
        let raw = b"\"Hello, World!\"";
        let mut json = JsonTokenStream::new(&raw[..]);

        assert_eq!(json.next().map(|t| t.to_owned()), Some(JsonToken::String("Hello, World!".into())));
        assert_eq!(json.next(), None);
    }

    #[test]
    fn stream_empty_string() {
        let raw = b"\"\"";
        let mut json = JsonTokenStream::new(&raw[..]);

        assert_eq!(json.next().map(|t| t.to_owned()), Some(JsonToken::String("".into())));
        assert_eq!(json.next(), None);
    }

    #[test]
    fn stream_number() {
        let raw = b"3.14e-1";
        let mut json = JsonTokenStream::new(&raw[..]);

        assert_eq!(json.next().map(|t| t.to_owned()), Some(JsonToken::Number(0.314)));
        assert_eq!(json.next(), None);
    }

    #[test]
    fn stream_number_ws() {
        let raw = b"3.14e-1 ";
        let mut json = JsonTokenStream::new(&raw[..]);

        assert_eq!(json.next().map(|t| t.to_owned()), Some(JsonToken::Number(0.314)));
        assert_eq!(json.next(), None);
    }

    #[test]
    fn stream_array() {
        let raw = b"[3.14, 2.71]";
        let mut json = JsonTokenStream::new(&raw[..]);

        assert_eq!(json.next(), Some(JsonToken::ArrayStart));
        assert_eq!(json.next().map(|t| t.to_owned()), Some(JsonToken::Number(3.14)));
        assert_eq!(json.next(), Some(JsonToken::Comma));
        assert_eq!(json.next().map(|t| t.to_owned()), Some(JsonToken::Number(2.71)));
        assert_eq!(json.next(), Some(JsonToken::ArrayEnd));
        assert_eq!(json.next(), None);
    }

    #[test]
    fn stream_object() {
        let raw = b"{\"a\": 1, \"b\": 2}";
        let mut json = JsonTokenStream::new(&raw[..]);

        assert_eq!(json.next(), Some(JsonToken::ObjectStart));
        assert_eq!(json.next().map(|t| t.to_owned()), Some(JsonToken::String("a".into())));
        assert_eq!(json.next(), Some(JsonToken::Colon));
        assert_eq!(json.next().map(|t| t.to_owned()), Some(JsonToken::Number(1.0)));
        assert_eq!(json.next(), Some(JsonToken::Comma));
        assert_eq!(json.next().map(|t| t.to_owned()), Some(JsonToken::String("b".into())));
        assert_eq!(json.next(), Some(JsonToken::Colon));
        assert_eq!(json.next().map(|t| t.to_owned()), Some(JsonToken::Number(2.0)));
        assert_eq!(json.next(), Some(JsonToken::ObjectEnd));
        assert_eq!(json.next(), None);
    }
}
