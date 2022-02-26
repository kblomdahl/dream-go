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

use dg_go::Board;

use super::sgf_token::*;
use super::stream::*;

use std::rc::Rc;

pub struct WithBoard<'a> {
    stream: Stream<'a>,
    board: Option<Rc<Board>>,
    komi: f32,
    include_all: bool
}

impl<'a> Stream<'a> {
    pub fn with_board(self) -> WithBoard<'a> {
        WithBoard {
            stream: self,
            board: None,
            komi: 7.5,
            include_all: false
        }
    }

    pub fn with_all_board(self, ) -> WithBoard<'a> {
        WithBoard {
            stream: self,
            board: None,
            komi: 7.5,
            include_all: true
        }
    }
}

impl<'a> WithBoard<'a> {
    fn ensure_board(board: &mut Option<Rc<Board>>, komi: f32) -> &mut Board {
        Rc::make_mut(board.get_or_insert_with(|| Rc::new(Board::new(komi))))
    }
}

impl<'a> Iterator for WithBoard<'a> {
    type Item = (Rc<Board>, SgfToken<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        for tok in &mut self.stream {
            match tok {
                SgfToken::Komi { .. } => {
                    let komi = tok.number();

                    if komi.is_finite() {
                        self.komi = komi;
                    }
                },
                SgfToken::Size { .. } if tok.number() != 19.0 => {
                    return None; // unsupported
                }
                SgfToken::Add { .. } => {
                    let board = Self::ensure_board(&mut self.board, self.komi);
                    let to_move = tok.color();
                    let point = tok.point();

                    if board.is_valid(to_move, point) {
                        board.place(to_move, point);
                    } else {
                        return None; // unsupported
                    }
                },
                SgfToken::Play { .. } => {
                    let board = Self::ensure_board(&mut self.board, self.komi);
                    let to_move = tok.color();
                    let point = tok.point();

                    if board.is_valid(to_move, point) {
                        board.place(to_move, point);
                    } else {
                        return None; // unsupported
                    }

                    return self.board.clone().map(|board| (board, tok));
                },
                _ => { /* pass */ }
            }

            if self.include_all && self.board.is_some() {
                return self.board.clone().map(|board| (board, tok));
            }
        }

        None
    }
}
