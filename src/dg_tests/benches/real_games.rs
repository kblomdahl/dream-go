// Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
extern crate dg_utils;
extern crate test;

use test::Bencher;

use dg_go::{DEFAULT_KOMI, Board, Color, Point};
use dg_go::utils::features::{HWC, DefaultFeatures, Features};
use dg_go::utils::symmetry::Transform;

/// Benchmark the full playout of a game as a serie of `is_valid` and `place` calls.
#[bench]
fn playout(b: &mut Bencher) {
    let rina_fujisawa_jeong_chio = [
        (Color::Black, 15,  3), (Color::White,  3,  3), (Color::Black, 15, 16), (Color::White,  3, 15),
        (Color::Black,  5,  2), (Color::White,  2,  5), (Color::Black, 16, 11), (Color::White, 13,  2),
        (Color::Black, 15,  5), (Color::White,  4,  2), (Color::Black, 14,  2), (Color::White, 13,  3),
        (Color::Black, 10,  3), (Color::White,  5,  3), (Color::Black, 12,  4), (Color::White, 12, 15),
        (Color::Black, 14, 15), (Color::White,  9, 16), (Color::Black,  9, 14), (Color::White, 12, 13),
        (Color::Black,  7, 15), (Color::White,  3, 13), (Color::Black, 11, 14), (Color::White, 12, 14),
        (Color::Black, 10, 16), (Color::White, 15, 12), (Color::Black, 15, 11), (Color::White, 14, 11),
        (Color::Black, 14, 12), (Color::White, 14, 14), (Color::Black, 15, 14), (Color::White, 14, 13),
        (Color::Black, 13, 12), (Color::White, 15, 13), (Color::Black, 16, 14), (Color::White, 14, 10),
        (Color::Black, 12, 11), (Color::White, 17, 13), (Color::Black, 17, 14), (Color::White, 10, 12),
        (Color::Black, 11, 10), (Color::White, 10, 15), (Color::Black,  9, 15), (Color::White, 10, 17),
        (Color::Black, 11, 16), (Color::White, 11, 15), (Color::Black, 11, 17), (Color::White,  8, 17),
        (Color::Black, 13, 16), (Color::White,  6, 16), (Color::Black,  3, 16), (Color::White,  4, 16),
        (Color::Black,  2, 15), (Color::White,  2, 16), (Color::Black,  3, 17), (Color::White,  2, 14),
        (Color::Black,  9, 11), (Color::White,  8, 13), (Color::Black,  1, 15), (Color::White,  4, 15),
        (Color::Black,  1, 16), (Color::White, 16,  7), (Color::Black, 15,  9), (Color::White, 13,  4),
        (Color::Black, 15,  7), (Color::White, 12,  5), (Color::Black, 10,  5), (Color::White, 10,  1),
        (Color::Black,  8,  2), (Color::White, 12,  1), (Color::Black, 11,  2), (Color::White, 11,  1),
        (Color::Black,  6,  2), (Color::White,  9,  2), (Color::Black,  8,  4), (Color::White,  9,  3),
        (Color::Black,  9,  4), (Color::White, 16,  2), (Color::Black, 14,  1), (Color::White, 16,  4),
        (Color::Black, 15,  4), (Color::White, 17,  5), (Color::Black,  8,  1), (Color::White, 12,  7),
        (Color::Black, 10,  7), (Color::White, 11,  4), (Color::Black, 10,  4), (Color::White, 15,  8),
        (Color::Black, 13,  6), (Color::White, 15,  6), (Color::Black, 16,  5), (Color::White, 14,  7),
        (Color::Black, 17,  6), (Color::White, 12,  6), (Color::Black, 17,  4), (Color::White, 14,  9),
        (Color::Black, 16,  9), (Color::White, 17,  7), (Color::Black, 16, 13), (Color::White,  4, 17),
        (Color::Black,  2, 17), (Color::White,  7, 11), (Color::Black, 12,  9), (Color::White, 12,  8),
        (Color::Black,  7,  9), (Color::White,  5,  9), (Color::Black,  2,  9), (Color::White,  2,  7),
        (Color::Black,  2, 12), (Color::White,  8,  7), (Color::Black,  6,  7), (Color::White,  8, 10),
        (Color::Black,  8,  9), (Color::White,  9, 10), (Color::Black, 10,  9), (Color::White,  6,  5),
        (Color::Black,  6,  4), (Color::White,  7,  6), (Color::Black, 10, 11), (Color::White,  8, 11),
        (Color::Black,  9, 12), (Color::White,  9, 13), (Color::Black, 10, 13), (Color::White, 10, 14),
        (Color::Black,  5,  5), (Color::White,  5,  6), (Color::Black,  6,  6), (Color::White,  7,  5),
        (Color::Black,  5,  4), (Color::White,  6,  8), (Color::Black,  5,  7), (Color::White,  4,  6),
        (Color::Black,  4,  7), (Color::White,  3,  7), (Color::Black,  4,  8), (Color::White,  9,  9),
        (Color::Black,  8, 12), (Color::White,  7, 12), (Color::Black, 11, 12), (Color::White,  8, 15),
        (Color::Black,  7,  7), (Color::White, 10,  8), (Color::Black,  9,  8), (Color::White,  8,  8),
        (Color::Black, 11,  8), (Color::White,  7,  8), (Color::Black,  8,  6), (Color::White, 12, 16),
        (Color::Black, 12, 17), (Color::White,  4, 10), (Color::Black,  3,  6), (Color::White,  4,  5),
        (Color::Black,  4,  4), (Color::White,  3,  5), (Color::Black,  1, 14), (Color::White,  5,  1),
        (Color::Black, 10, 18), (Color::White,  9, 17), (Color::Black,  3, 12), (Color::White,  4, 12),
        (Color::Black,  6,  1), (Color::White,  4,  1), (Color::Black,  3,  4), (Color::White,  2,  4),
        (Color::Black,  4,  3), (Color::White,  2,  2), (Color::Black,  1,  8), (Color::White, 13,  1),
        (Color::Black, 16, 12), (Color::White, 15,  1), (Color::Black, 14,  0), (Color::White, 17,  3),
        (Color::Black, 16,  3), (Color::White, 17,  1), (Color::Black, 15,  0), (Color::White, 15,  2),
        (Color::Black, 14,  3), (Color::White, 16,  0), (Color::Black, 18,  3), (Color::White, 17,  2),
        (Color::Black,  0,  5), (Color::White,  0,  4), (Color::Black, 14,  5), (Color::White, 16,  6),
        (Color::Black, 18,  6), (Color::White, 18,  7), (Color::Black, 18,  5), (Color::White,  3, 11),
        (Color::Black,  2, 11), (Color::White,  2, 13), (Color::Black,  1, 13), (Color::White,  3,  8),
        (Color::Black,  8,  5), (Color::White,  4,  9), (Color::Black,  7,  4), (Color::White, 11, 13),
        (Color::Black,  1,  7), (Color::White,  1,  6), (Color::Black,  0,  6), (Color::White,  0,  7),
        (Color::Black,  0,  8), (Color::White,  1,  5), (Color::Black,  3,  9), (Color::White,  3, 10),
        (Color::Black,  4, 18), (Color::White,  5, 18), (Color::Black,  3, 18), (Color::White,  2, 10),
        (Color::Black,  1, 10), (Color::White,  4, 13), (Color::Black,  0,  7), (Color::White,  2,  6),
        (Color::Black,  1,  3), (Color::White,  1,  4), (Color::Black,  2,  3), (Color::White,  3,  2),
        (Color::Black,  0,  3), (Color::White,  1,  2), (Color::Black,  9,  1), (Color::White, 10,  2),
        (Color::Black, 12,  3), (Color::White, 11,  5), (Color::Black, 11,  3), (Color::White, 13,  5),
        (Color::Black, 14,  6), (Color::White, 16,  8), (Color::Black, 17,  9), (Color::White,  9,  7),
        (Color::Black,  9,  6), (Color::White, 13,  0), (Color::Black,  7, 17), (Color::White,  6, 18)
    ];

    b.iter(|| {
        let mut board = Board::new(DEFAULT_KOMI);

        for &(color, x, y) in rina_fujisawa_jeong_chio.iter() {
            debug_assert!(board.is_valid(color, Point::new(x, y)));

            if board.is_valid(color, Point::new(x, y)) {
                board.place(color, Point::new(x, y));
            }
        }

        board
    });
}

/// Benchmark feature extraction from a given board position in the `f16` data type.
#[bench]
fn get_features_16(b: &mut Bencher) {
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

    let mut board = Board::new(DEFAULT_KOMI);

    for &(color, x, y) in lee_sedol_alphago_4_78.iter() {
        assert!(board.is_valid(color, Point::new(x, y)));

        board.place(color, Point::new(x, y));
    }

    b.iter(move || {
        let black = test::black_box(Color::Black);

        DefaultFeatures::new(&board).get_features::<HWC, f32>(black, Transform::Transpose)
    });
}

/// Benchmark feature extraction from a given board position in the `f32` data type.
#[bench]
fn get_features_32(b: &mut Bencher) {
    let rina_fujisawa_zhiying_yu = [
        (Color::Black, 15,  3), (Color::White,  3,  2), (Color::Black,  3, 16), (Color::White, 16, 16),
        (Color::Black,  2,  4), (Color::White,  3, 14), (Color::Black,  6,  2), (Color::White,  4,  3),
        (Color::Black,  6,  4), (Color::White,  4,  5), (Color::Black,  2,  7), (Color::White,  2, 16),
        (Color::Black,  1,  2), (Color::White,  4,  7), (Color::Black,  3,  9), (Color::White, 10,  3),
        (Color::Black,  9,  4), (Color::White, 10,  4), (Color::Black,  9,  5), (Color::White,  9,  3),
        (Color::Black,  3,  6), (Color::White,  4,  6), (Color::Black,  7,  5), (Color::White,  6,  8),
        (Color::Black, 13,  2), (Color::White, 11,  6), (Color::Black,  9,  7), (Color::White, 11,  8),
        (Color::Black,  9,  9), (Color::White,  6, 10), (Color::Black, 11, 10), (Color::White, 13,  9),
        (Color::Black, 13, 11), (Color::White, 17,  3), (Color::Black, 16,  5), (Color::White, 17,  5),
        (Color::Black, 16,  6), (Color::White, 16,  2), (Color::Black, 15,  2), (Color::White, 16,  4),
        (Color::Black, 15,  4), (Color::White, 16,  1), (Color::Black, 13,  6), (Color::White, 12,  3),
        (Color::Black, 15,  1), (Color::White, 10, 10), (Color::Black, 10,  9), (Color::White, 11,  9),
        (Color::Black, 11, 11), (Color::White,  7,  3), (Color::Black,  6,  3), (Color::White, 10, 11),
        (Color::Black, 10, 12), (Color::White,  8, 10), (Color::Black, 10,  7), (Color::White, 11,  7),
        (Color::Black,  9, 10), (Color::White,  9, 11), (Color::Black,  8, 11), (Color::White,  9, 12),
        (Color::Black,  8, 12), (Color::White,  9, 13), (Color::Black,  7, 10), (Color::White,  8, 13),
        (Color::Black,  6, 11), (Color::White,  4, 10), (Color::Black, 15, 15), (Color::White, 15, 16),
        (Color::Black, 13, 15), (Color::White, 14, 15), (Color::Black, 14, 14), (Color::White, 14, 16),
        (Color::Black, 16,  3), (Color::White, 13, 14), (Color::Black, 15, 10), (Color::White, 12, 14),
        (Color::Black, 15, 12), (Color::White, 14, 10), (Color::Black, 15, 11), (Color::White, 17,  4),
        (Color::Black, 14,  8), (Color::White, 17,  6), (Color::Black, 17,  7), (Color::White, 15,  0),
        (Color::Black,  4,  9), (Color::White,  2,  5), (Color::Black,  3,  5), (Color::White,  3,  4),
        (Color::Black,  1,  5), (Color::White,  2,  3), (Color::Black,  1,  4), (Color::White,  5,  9),
        (Color::Black,  5, 11), (Color::White,  5, 10), (Color::Black, 11, 14), (Color::White, 11, 13),
        (Color::Black, 14,  0), (Color::White, 16,  0), (Color::Black, 11,  2), (Color::White, 10,  6),
        (Color::Black,  9,  6), (Color::White, 13,  5), (Color::Black, 14,  5), (Color::White,  7,  1),
        (Color::Black,  6,  1), (Color::White, 11,  1), (Color::Black, 10,  1), (Color::White, 10,  2),
        (Color::Black, 12,  1), (Color::White,  9,  1), (Color::Black, 11,  0), (Color::White,  8,  2),
        (Color::Black, 13, 13), (Color::White, 12, 13), (Color::Black,  3, 13), (Color::White,  3, 15),
        (Color::Black,  4, 13), (Color::White,  4, 16), (Color::Black,  6,  6), (Color::White,  6,  7),
        (Color::Black,  4,  8), (Color::White,  5,  8), (Color::Black,  2,  9), (Color::White,  1,  1),
        (Color::Black,  6,  0), (Color::White, 10,  5), (Color::Black,  2,  1), (Color::White,  2,  2),
        (Color::Black,  0,  1), (Color::White,  7, 12), (Color::Black,  7, 11), (Color::White,  7,  9),
        (Color::Black,  8,  9), (Color::White,  4,  1), (Color::Black,  5,  2)
    ];

    let mut board = Board::new(DEFAULT_KOMI);

    for &(color, x, y) in rina_fujisawa_zhiying_yu.iter() {
        assert!(board.is_valid(color, Point::new(x, y)));

        board.place(color, Point::new(x, y));
    }

    b.iter(move || {
        let white = test::black_box(Color::White);

        DefaultFeatures::new(&board).get_features::<HWC, f32>(white, Transform::FlipLR)
    });
}
