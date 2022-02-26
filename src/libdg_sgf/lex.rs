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

use super::lex_token::*;

pub struct Lex<'a> {
    input: &'a [u8],
    offset: usize
}

impl<'a> Lex<'a> {
    pub fn lex(s: &'a [u8]) -> Self {
        Self {
            input: s,
            offset: 0
        }
    }

    fn try_consume(&mut self, byte: u8) -> bool {
        let is_match = self.input[self.offset] == byte;
        self.offset += is_match as usize;
        is_match
    }

    fn is_control(byte: u8) -> bool {
        byte == b'[' || byte == b']' || byte == b'(' || byte == b')' || byte == b';'
    }

    fn try_consume_text(&mut self) -> Option<usize> {
        let original_offset = self.offset;

        while self.offset < self.input.len() {
            if self.input[self.offset] == b'\\' {
                self.offset += 2;
            } else if !Self::is_control(self.input[self.offset]) {
                self.offset += 1;
            } else {
                break
            }
        }

        if self.offset > original_offset {
            Some(self.offset - original_offset)
        } else {
            None
        }
    }

    fn skip_ws(&mut self) -> Option<usize> {
        while self.offset < self.input.len() {
            if !self.input[self.offset].is_ascii_whitespace() {
                return Some(self.offset);
            }
            self.offset += 1;
        }

        None
    }
}

impl<'a> Iterator for Lex<'a> {
    type Item = LexToken;

    fn next(&mut self) -> Option<Self::Item> {
        self.skip_ws().and_then(|offset| {
            if self.try_consume(b'[') {
                Some(LexToken::LBrack)
            } else if self.try_consume(b']') {
                Some(LexToken::RBrack)
            } else if self.try_consume(b'(') {
                Some(LexToken::LParen)
            } else if self.try_consume(b')') {
                Some(LexToken::RParen)
            } else if self.try_consume(b';') {
                Some(LexToken::SemiColon)
            } else if let Some(len) = self.try_consume_text() {
                Some(LexToken::Text { offset, len })
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lbrack() {
        assert_eq!(
            Lex::lex(b"[").collect::<Vec<_>>(),
            vec! [ LexToken::LBrack ]
        );
    }

    #[test]
    fn rbrack() {
        assert_eq!(
            Lex::lex(b"]").collect::<Vec<_>>(),
            vec! [ LexToken::RBrack ]
        );
    }

    #[test]
    fn lparen() {
        assert_eq!(
            Lex::lex(b"(").collect::<Vec<_>>(),
            vec! [ LexToken::LParen ]
        );
    }

    #[test]
    fn rparen() {
        assert_eq!(
            Lex::lex(b")").collect::<Vec<_>>(),
            vec! [ LexToken::RParen ]
        );
    }

    #[test]
    fn semi() {
        assert_eq!(
            Lex::lex(b";").collect::<Vec<_>>(),
            vec! [ LexToken::SemiColon ]
        );
    }

    #[test]
    fn text() {
        assert_eq!(
            Lex::lex(b"Hello, World!").collect::<Vec<_>>(),
            vec! [ LexToken::Text { offset: 0, len: 13 } ]
        );
    }

    #[test]
    fn game_tree() {
        assert_eq!(
            Lex::lex(b"(;GM[3])").collect::<Vec<_>>(),
            vec! [
                LexToken::LParen,
                LexToken::SemiColon,
                LexToken::Text { offset: 2, len: 2 },
                LexToken::LBrack,
                LexToken::Text { offset: 5, len: 1 },
                LexToken::RBrack,
                LexToken::RParen
            ]
        );
    }

    #[test]
    fn escape_sequence() {
        assert_eq!(
            Lex::lex(b"(;C[A\\]])").collect::<Vec<_>>(),
            vec! [
                LexToken::LParen,
                LexToken::SemiColon,
                LexToken::Text { offset: 2, len: 1 },
                LexToken::LBrack,
                LexToken::Text { offset: 4, len: 3 },
                LexToken::RBrack,
                LexToken::RParen
            ]
        );
    }
}
