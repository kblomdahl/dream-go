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

use dg_go::{Board};

use super::sgf_token::*;
use super::stream::*;

use std::rc::Rc;

struct BoardState {
    board: Option<Rc<Board>>,
    komi: f32,
    is_resign: bool
}

impl BoardState {
    fn empty() -> Self {
        Self {
            board: None,
            komi: 7.5,
            is_resign: false
        }
    }

    fn ensure_board(board: &mut Option<Rc<Board>>, komi: f32) -> &mut Board {
        Rc::make_mut(board.get_or_insert_with(|| Rc::new(Board::new(komi))))
    }

    fn process_token<'a>(&mut self, tok: &SgfToken<'a>) -> bool {
        match tok {
            SgfToken::Result { text } => {
                self.is_resign = text.windows(2).any(|w| w == b"+R");
            },
            SgfToken::Komi { .. } => {
                let komi = tok.number();

                if komi.is_finite() {
                    self.komi = komi;
                } else {
                    return false; // unsupported
                }
            },
            SgfToken::Size { .. } if tok.number() != 19.0 => {
                return false; // unsupported
            }
            SgfToken::Add { .. } => {
                let board = Self::ensure_board(&mut self.board, self.komi);
                let to_move = tok.color();
                let point = tok.point();

                if board.is_valid(to_move, point) {
                    board.place(to_move, point);
                } else {
                    return false; // unsupported
                }
            },
            SgfToken::Play { .. } => {
                let board = Self::ensure_board(&mut self.board, self.komi);
                let to_move = tok.color();
                let point = tok.point();

                if board.is_valid(to_move, point) {
                    board.place(to_move, point);
                } else {
                    return false; // unsupported
                }
            },
            _ => { /* pass */ }
        }

        true
    }
}

pub struct WithBoard<'a> {
    stream: Stream<'a>,
    buf: Vec<SgfToken<'a>>,
    state: BoardState
}

pub struct OnlyBoard<'a> {
    stream: Stream<'a>,
    buf: Vec<SgfToken<'a>>,
    state: BoardState
}

impl<'a> Stream<'a> {
    pub fn only_board(self) -> OnlyBoard<'a> {
        OnlyBoard {
            stream: self,
            buf: vec! [],
            state: BoardState::empty()
        }
    }

    pub fn with_board(self, ) -> WithBoard<'a> {
        WithBoard {
            stream: self,
            buf: vec! [],
            state: BoardState::empty()
        }
    }
}

impl<'a> WithBoard<'a> {
    pub fn board(&self) -> &Option<Rc<Board>> {
        &self.state.board
    }

    pub fn komi(&self) -> f32 {
        self.state.komi
    }

    pub fn is_resign(&self) -> bool {
        self.state.is_resign
    }
}

fn try_fill_buf<'a>(buf: &mut Vec<SgfToken<'a>>, from_stream: &mut Stream<'a>) -> bool {
    if !buf.is_empty() {
        return true;
    }

    for tok in from_stream {
        let is_node = tok == SgfToken::Node;
        buf.push(tok);

        if is_node {
            break
        }
    }

    buf.reverse();
    !buf.is_empty()
}

impl<'a> Iterator for WithBoard<'a> {
    type Item = (Rc<Board>, SgfToken<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !try_fill_buf(&mut self.buf, &mut self.stream) {
                return None;
            }

            while let Some(tok) = self.buf.pop() {
                if !self.state.process_token(&tok) {
                    return None;
                }

                BoardState::ensure_board(&mut self.state.board, self.state.komi);
                return Some((self.state.board.as_ref().unwrap().clone(), tok));
            }
        }
    }
}

impl<'a> Iterator for OnlyBoard<'a> {
    type Item = Rc<Board>;

    fn next(&mut self) -> Option<Self::Item> {
        if !try_fill_buf(&mut self.buf, &mut self.stream) {
            return None;
        }

        while let Some(tok) = self.buf.pop() {
            if !self.state.process_token(&tok) {
                return None;
            }
        }

        BoardState::ensure_board(&mut self.state.board, self.state.komi);
        self.state.board.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CGoban, ToSgf};

    const SGF: &[u8] = b"(;GM[1]FF[4]RE[W+Resign]KM[0.5];B[dd];W[pp])";

    #[test]
    fn only_include_empty_board() {
        let stream = Stream::new(SGF);
        let boards = stream.only_board().map(|board| board.to_sgf::<CGoban>()).collect::<Vec<_>>();

        assert_eq!(
            boards,
            vec! [
                "(;)",
                "(;)",
                "(;AB[dd])",
                "(;AB[dd]AW[pp])",
            ]
        );
    }

    #[test]
    fn with_include_empty_board() {
        let stream = Stream::new(SGF);
        let boards = stream.with_board().map(|(board, tok)| (board.to_sgf::<CGoban>(), tok)).collect::<Vec<_>>();

        assert_eq!(
            boards,
            vec! [
                ("(;)".into(), SgfToken::Node),
                ("(;)".into(), SgfToken::Result { text: b"W+Resign" }),
                ("(;)".into(), SgfToken::Komi { text: b"0.5" }),
                ("(;)".into(), SgfToken::Node),
                ("(;AB[dd])".into(), SgfToken::Play { color: b"B", point: b"dd" }),
                ("(;AB[dd])".into(), SgfToken::Node),
                ("(;AB[dd]AW[pp])".into(), SgfToken::Play { color: b"W", point: b"pp" }),
            ]
        );
    }
}
