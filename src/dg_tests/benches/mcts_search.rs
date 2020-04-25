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
#![feature(test)]

extern crate dg_go;
extern crate dg_mcts;
extern crate test;

use dg_go::{Board, Color, Point};
use dg_mcts::time_control::RolloutLimit;
use dg_mcts as mcts;
use test::Bencher;
use dg_mcts::options::StandardSearch;

#[bench]
fn lee_sedol_alphago_4_78(b: &mut Bencher) {
    let lee_sedol_alphago_4_78 = [
        (Color::Black, 15,  3), (Color::White,  3, 15), (Color::Black,  2,  3), (Color::White, 16, 15),
        (Color::Black, 14, 15), (Color::White, 14, 16), (Color::Black, 13, 16), (Color::White, 15, 16),
        (Color::Black,  2, 13), (Color::White,  5, 16), (Color::Black, 12, 15), (Color::White, 15, 14),
        (Color::Black,  8, 16), (Color::White,  4,  2), (Color::Black,  7,  3), (Color::White,  2,  6),
        (Color::Black,  4,  3), (Color::White,  2,  9), (Color::Black,  3,  2), (Color::White,  1, 15),
        (Color::Black, 13,  2), (Color::White, 16,  8), (Color::Black,  4, 15), (Color::White,  4, 14),
        (Color::Black,  3, 10), (Color::White,  5, 15), (Color::Black,  2, 10), (Color::White,  3,  9),
        (Color::Black,  4,  9), (Color::White,  4,  8), (Color::Black,  5,  8), (Color::White,  4,  7),
        (Color::Black,  5,  7), (Color::White,  1,  9), (Color::Black,  5, 10), (Color::White,  5,  6),
        (Color::Black,  6,  6), (Color::White,  5,  5), (Color::Black,  6,  5), (Color::White, 12,  2),
        (Color::Black, 12,  3), (Color::White, 11,  2), (Color::Black, 13,  1), (Color::White,  8,  3),
        (Color::Black,  7,  2), (Color::White,  9,  6), (Color::Black, 15,  9), (Color::White, 15,  8),
        (Color::Black, 14,  9), (Color::White, 14,  8), (Color::Black, 13,  8), (Color::White, 13,  7),
        (Color::Black, 12,  7), (Color::White, 13,  6), (Color::Black, 12,  6), (Color::White, 12,  8),
        (Color::Black, 13,  9), (Color::White, 12,  5), (Color::Black, 11,  8), (Color::White, 13,  4),
        (Color::Black, 13,  3), (Color::White, 12,  9), (Color::Black, 11,  5), (Color::White, 12, 10),
        (Color::Black, 12,  4), (Color::White, 13,  5), (Color::Black, 11,  7), (Color::White, 16,  9),
        (Color::Black, 10, 10), (Color::White,  8, 10), (Color::Black,  9,  8), (Color::White,  6,  7),
        (Color::Black,  7,  9), (Color::White,  6,  4), (Color::Black,  7,  4), (Color::White,  5,  3),
        (Color::Black,  5,  2), (Color::White, 10,  8)
    ];

    let mut original_board = Board::new(7.5);

    for &(color, x, y) in lee_sedol_alphago_4_78.iter() {
        assert!(original_board.is_valid(color, Point::new(x, y)));

        original_board.place(color, Point::new(x, y));
    }

    b.iter(move || {
        let server = mcts::predict::RandomPredictor::default();

        mcts::predict::<_, _, StandardSearch>(
            &server,
            Some(4),
            RolloutLimit::new(40),
            None,
            &original_board,
            Color::Black
        )
    });
}