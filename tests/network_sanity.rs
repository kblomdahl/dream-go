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
//
extern crate dream_go;
extern crate ordered_float;

use dream_go::go::symmetry::Transform;
use dream_go::go::{Board, Color, CHW, HWC};
use dream_go::nn;
use dream_go::util::types::*;
use ordered_float::OrderedFloat;

thread_local! {
    static NETWORK: nn::Network = nn::Network::new().unwrap();
}

fn predict(moves: &[(Color, usize, usize)], next_color: Color) -> (f32, Box<[f32]>) {
    // place each stone on a fresh board to re-produce the board state
    let mut board = Board::new();

    for (i, &(color, x, y)) in moves.iter().enumerate() {
        assert!(board.is_valid(color, x, y), "move {} is not valid ({}, {})", i, x, y);

        board.place(color, x, y);
    }

    // predict the next move and the current value using the neural network
    NETWORK.with(|network| {
        let mut workspace = network.get_workspace(1);

        match *nn::TYPE {
            nn::Type::Single => {
                let features = board.get_features::<f32, CHW>(next_color, Transform::Identity);
                let (value, policy) = nn::forward::<f32, f32>(&mut workspace, &vec! [features]);

                (value[0], policy[0].clone())
            },
            nn::Type::Half => {
                let features = board.get_features::<f16, CHW>(next_color, Transform::Identity);
                let (value, policy) = nn::forward::<f16, f16>(&mut workspace, &vec! [features]);
                let policy = policy[0].iter()
                    .map(|&p| f32::from(p))
                    .collect::<Vec<f32>>();

                (f32::from(value[0]), policy.into_boxed_slice())
            },
            nn::Type::Int8 => {
                let features = board.get_features::<q8, HWC>(next_color, Transform::Identity);
                let (value, policy) = nn::forward::<q8, f32>(&mut workspace, &vec! [features]);

                (value[0], policy[0].clone())
            }
        }
    })
}

/// Returns the vertex with the maximum value in the given policy.
///
/// # Arguments
///
/// * `policy` -
///
fn policy_to_vertex(policy: &[f32]) -> (usize, usize) {
    assert_eq!(policy.len(), 362);

    if let Some(index) = (0..362).max_by_key(|&i| OrderedFloat(policy[i])) {
        if index == 362 {
            (19, 19)  // pass
        } else {
            let y = index / 19;
            let x = index % 19;

            (x, y)
        }
    } else {
        (19, 19)  // pass
    }
}

/// Test that the engine does not try to play out a ladder that does
/// not work.
#[test]
fn ladder_1() {
    let moves = [
        (Color::Black,  3, 16), (Color::White, 15,  3), (Color::Black,  3, 14), (Color::White,  3,  3),
        (Color::Black, 15, 15), (Color::White,  3,  2), (Color::Black, 16,  2), (Color::White, 16,  3),
        (Color::Black, 15,  2), (Color::White, 14,  2), (Color::Black, 14,  1), (Color::White, 13,  2),
        (Color::Black, 13,  1), (Color::White, 12,  2), (Color::Black, 12,  1), (Color::White, 11,  2),
        (Color::Black, 11,  1), (Color::White,  9,  2), (Color::Black, 16, 13), (Color::White,  3, 13),
        (Color::Black,  2, 13), (Color::White,  2, 12), (Color::Black,  2, 14), (Color::White,  4, 13),
        (Color::Black,  7,  7), (Color::White,  3,  9), (Color::Black,  1, 12), (Color::White,  2, 11),
        (Color::Black,  1, 11), (Color::White,  1, 10), (Color::Black,  4,  7), (Color::White,  2,  7),
        (Color::Black,  3,  5), (Color::White,  4,  6), (Color::Black,  3,  6), (Color::White,  3,  7),
        (Color::Black,  4,  5), (Color::White,  5,  6), (Color::Black,  5,  5), (Color::White,  6,  6),
        (Color::Black,  7,  5), (Color::White,  6,  5), (Color::Black,  6,  4), (Color::White,  7,  4),
        (Color::Black,  7,  6), (Color::White,  5,  7), (Color::Black,  6,  3), (Color::White,  8,  4),
        (Color::Black,  8,  2), (Color::White,  7,  2), (Color::Black,  7,  3), (Color::White,  8,  3),
        (Color::Black,  8,  1), (Color::White,  7,  1), (Color::Black,  9,  1), (Color::White, 10,  1),
        (Color::Black,  9,  3), (Color::White, 10,  2), (Color::Black,  9,  4), (Color::White,  8,  5),
        (Color::Black,  4,  8), (Color::White,  5,  8), (Color::Black,  4,  9), (Color::White,  5,  9),
        (Color::Black,  4, 10), (Color::White,  5, 10), (Color::Black,  4, 11), (Color::White,  5, 11),
        (Color::Black,  9,  5)
    ];

    // white should not (!) play `(Color::White,  8,  6)`
    let (_value, policy) = predict(&moves, Color::White);
    let (x, y) = policy_to_vertex(&policy);

    assert!(x != 8 || y != 6, "Broken ladder at (8, 6) -- ({}, {})", x, y);
}

/// Test that the engine does not try to play out a ladder that does
/// not work.
#[test]
fn ladder_2() {
    let moves = [
        (Color::Black, 15,  3), (Color::White,  3,  3), (Color::Black, 15, 15), (Color::White,  3, 15),
        (Color::Black, 15, 12), (Color::White,  5, 16), (Color::Black,  4, 16), (Color::White,  3, 16),
        (Color::Black,  5, 15), (Color::White,  4, 15), (Color::Black,  4, 17), (Color::White,  6, 16),
        (Color::Black,  6, 15), (Color::White,  7, 15), (Color::Black,  7, 16), (Color::White,  6, 14),
        (Color::Black,  3, 17), (Color::White,  2, 17)
    ];

    // black should not (!) play `(Color::Black,  5, 14)`
    let (_value, policy) = predict(&moves, Color::Black);
    let (x, y) = policy_to_vertex(&policy);

    assert!(x != 5 || y != 14, "Broken ladder at (5, 14) -- ({}, {})", x, y);
}

/// Test that the engine does not try to play out a ladder that does
/// not work.
#[test]
fn ladder_3() {
    // pass
}

/// Test that the engine correctly detects that a large black dragon is
/// dead.
#[test]
fn dead_dragon_1() {
    let moves = [
        (Color::Black, 15,  3), (Color::White, 15, 15), (Color::Black,  3,  3), (Color::White,  3, 15),
        (Color::Black, 13, 16), (Color::White, 16, 13), (Color::Black, 15, 17), (Color::White, 16, 16),
        (Color::Black, 10, 16), (Color::White, 13,  2), (Color::Black, 11,  2), (Color::White, 16,  5),
        (Color::Black, 13,  3), (Color::White, 16,  2), (Color::Black, 16,  3), (Color::White, 15,  2),
        (Color::Black, 14,  2), (Color::White, 14,  1), (Color::Black, 14,  3), (Color::White, 13,  1),
        (Color::Black, 12,  2), (Color::White, 17,  3), (Color::Black, 17,  4), (Color::White, 17,  2),
        (Color::Black, 16,  4), (Color::White, 17,  5), (Color::Black, 14,  5), (Color::White, 15,  7),
        (Color::Black, 13,  7), (Color::White, 15,  9), (Color::Black,  6,  7), (Color::White,  5,  2),
        (Color::Black,  7,  2), (Color::White,  2,  2), (Color::Black,  3,  2), (Color::White,  2,  3),
        (Color::Black,  3,  4), (Color::White,  3,  1), (Color::Black,  4,  1), (Color::White,  2,  1),
        (Color::Black,  4,  2), (Color::White,  2,  5), (Color::Black,  2, 13), (Color::White,  5, 16),
        (Color::Black,  1, 15), (Color::White,  2, 16), (Color::Black,  3, 11), (Color::White,  3,  5),
        (Color::Black,  4,  5), (Color::White,  4,  6), (Color::Black,  5,  5), (Color::White,  2,  9),
        (Color::Black,  3,  6), (Color::White,  2,  6), (Color::Black,  3,  7), (Color::White,  2,  7),
        (Color::Black,  3,  8), (Color::White,  3,  9), (Color::Black,  7, 16), (Color::White,  1, 16),
        (Color::Black,  5, 17), (Color::White,  4, 17), (Color::Black,  5, 15), (Color::White,  6, 16),
        (Color::Black,  6, 15), (Color::White,  7, 15), (Color::Black,  7, 14), (Color::White,  8, 15),
        (Color::Black,  8, 16), (Color::White,  9, 15), (Color::Black,  9, 16), (Color::White,  8, 13),
        (Color::Black,  7, 12), (Color::White,  9, 11), (Color::Black,  7, 13), (Color::White, 10, 13),
        (Color::Black,  8, 14), (Color::White,  9, 14), (Color::Black,  8, 12), (Color::White,  9, 12),
        (Color::Black,  8, 10), (Color::White, 10,  9), (Color::Black, 13,  9), (Color::White, 12,  8),
        (Color::Black, 13,  8), (Color::White, 10,  7), (Color::Black, 11, 11), (Color::White,  9, 10),
        (Color::Black,  9,  8), (Color::White, 10,  8), (Color::Black,  9,  7), (Color::White, 12, 10),
        (Color::Black, 13, 10), (Color::White, 12, 11), (Color::Black, 15, 10), (Color::White, 16, 10),
        (Color::Black, 16, 11), (Color::White, 15, 11), (Color::Black, 14, 10), (Color::White, 16,  9),
        (Color::Black, 15, 12), (Color::White, 17, 11), (Color::Black, 16, 12), (Color::White, 17, 12),
        (Color::Black, 15, 13), (Color::White, 15, 14), (Color::Black, 13, 14), (Color::White, 14, 13),
        (Color::Black, 14, 12), (Color::White, 13, 12), (Color::Black, 14, 11), (Color::White, 12, 15),
        (Color::Black, 13, 15), (Color::White, 13, 13), (Color::Black, 12, 14), (Color::White, 10,  5),
        (Color::Black,  9,  6), (Color::White, 10,  6), (Color::Black, 12,  9), (Color::White, 11, 12),
        (Color::Black, 12, 13), (Color::White, 12, 12), (Color::Black, 14, 14), (Color::White, 12,  5),
        (Color::Black, 12,  7), (Color::White, 14,  6), (Color::Black, 13,  6), (Color::White, 13,  5),
        (Color::Black, 15,  5), (Color::White, 15,  6), (Color::Black, 11,  9), (Color::White, 11, 10),
        (Color::Black, 11,  8), (Color::White, 12,  3), (Color::Black, 12,  4), (Color::White, 11,  3),
        (Color::Black, 11,  4), (Color::White, 10,  3), (Color::Black, 10,  4), (Color::White, 13,  4),
        (Color::Black,  9,  3), (Color::White, 10,  2), (Color::Black, 10,  1), (Color::White,  9,  2),
        (Color::Black,  8,  2), (Color::White, 14,  4), (Color::Black, 11,  5), (Color::White, 11,  6),
        (Color::Black,  9,  1), (Color::White, 12,  6), (Color::Black, 15,  4), (Color::White, 18,  4),
        (Color::Black, 12,  3), (Color::White,  9,  5), (Color::Black, 10, 11), (Color::White, 10, 15),
        (Color::Black, 11, 15), (Color::White, 11, 16), (Color::Black, 11, 14), (Color::White,  9, 13),
        (Color::Black,  6, 17), (Color::White,  4, 16), (Color::Black,  8, 11), (Color::White, 11, 17),
        (Color::Black, 10, 17), (Color::White,  5, 18), (Color::Black,  7, 17), (Color::White,  3, 13),
        (Color::Black,  6, 14), (Color::White,  3, 12), (Color::Black,  8,  6), (Color::White,  4, 11),
        (Color::Black,  8,  9), (Color::White, 16, 17), (Color::Black, 14, 17), (Color::White,  4,  8),
        (Color::Black,  4,  7), (Color::White,  5,  8), (Color::Black,  5,  7), (Color::White,  8,  5),
        (Color::Black,  7,  5), (Color::White,  4,  0), (Color::Black,  5,  0), (Color::White,  3,  0),
        (Color::Black,  5,  1), (Color::White,  5, 10), (Color::Black,  4, 10), (Color::White,  5, 12),
        (Color::Black,  2, 12), (Color::White,  1, 11), (Color::Black,  2, 11), (Color::White,  1, 10),
        (Color::Black,  2,  8), (Color::White,  1,  8), (Color::Black,  3, 10), (Color::White,  4,  9),
        (Color::Black,  6,  8), (Color::White, 12,  1), (Color::Black, 11, 18), (Color::White,  8,  4),
        (Color::Black,  7,  4), (Color::White,  8,  3), (Color::Black,  7,  3), (Color::White, 15, 18),
        (Color::Black, 14, 18), (Color::White, 16, 18), (Color::Black, 12, 17), (Color::White,  6, 18),
        (Color::Black,  7, 18), (Color::White,  4, 18), (Color::Black, 10, 18), (Color::White, 11,  0),
        (Color::Black, 11,  1), (Color::White,  6,  9), (Color::Black,  6, 11), (Color::White,  5, 11),
        (Color::Black,  5, 13), (Color::White,  4, 14), (Color::Black,  6, 12), (Color::White,  7,  9),
        (Color::Black,  7,  8), (Color::White,  7, 10), (Color::Black,  9,  9), (Color::White, 10, 10),
        (Color::Black, 11,  7), (Color::White, 14,  8), (Color::Black,  9,  4), (Color::White, 14,  9),
        (Color::Black, 17, 13), (Color::White, 16, 14), (Color::Black, 17, 14), (Color::White, 14,  7),
        (Color::Black, 17, 15), (Color::White, 17, 16), (Color::Black, 16, 15), (Color::White, 18, 15),
        (Color::Black, 15, 16), (Color::White, 18, 13), (Color::Black, 14, 15), (Color::White, 18, 14),
        (Color::Black, 18, 12), (Color::White, 18, 11), (Color::Black, 17, 10), (Color::White, 17,  9),
        (Color::Black, 18,  3), (Color::White, 18,  2), (Color::Black, 18, 10), (Color::White, 18,  9),
        (Color::Black, 17,  7), (Color::White, 16,  6), (Color::Black, 17,  8), (Color::White, 12,  0),
        (Color::Black, 10,  0), (Color::White,  2,  4), (Color::Black,  5,  3), (Color::White,  5, 14),
        (Color::Black,  2, 14), (Color::White,  1, 12), (Color::Black,  1, 13), (Color::White,  0, 16),
        (Color::Black,  0, 13), (Color::White,  0, 15), (Color::Black,  0, 14), (Color::White,  2, 15),
        (Color::Black,  1, 14)
    ];

    // black should win by 38.5 points
    let (value, _policy) = predict(&moves, Color::Black);

    assert!(value > 0.0, "Black should win by 38.5 -- {}", value);
}

/// Test that the engine correctly detects that white has won the game.
#[test]
fn end_1() {
    let moves = [
        (Color::Black,  3, 16), (Color::White,  2,  3), (Color::Black, 16, 15), (Color::White, 15,  3),
        (Color::Black,  2, 13), (Color::White, 14, 15), (Color::Black, 14, 16), (Color::White, 13, 16),
        (Color::Black, 15, 16), (Color::White, 13, 15), (Color::Black, 16, 13), (Color::White,  9, 16),
        (Color::Black,  7, 16), (Color::White,  4,  2), (Color::Black, 11, 16), (Color::White, 11, 15),
        (Color::Black, 11, 17), (Color::White, 15, 15), (Color::Black, 16, 16), (Color::White, 10, 15),
        (Color::Black, 13, 17), (Color::White, 14, 12), (Color::Black, 13,  2), (Color::White, 10,  2),
        (Color::Black, 16,  2), (Color::White, 15,  2), (Color::Black, 16,  3), (Color::White, 15,  4),
        (Color::Black, 15,  1), (Color::White, 14,  1), (Color::Black, 16,  1), (Color::White, 14,  2),
        (Color::Black, 16,  5), (Color::White,  2,  8), (Color::Black,  4,  4), (Color::White,  2, 11),
        (Color::Black,  2,  4), (Color::White,  3,  3), (Color::Black, 17, 18), (Color::White,  3,  4),
        (Color::Black,  9,  3), (Color::White, 10,  3), (Color::Black,  9,  4), (Color::White, 10,  4),
        (Color::Black,  1,  3), (Color::White,  2,  5), (Color::Black,  9,  5), (Color::White,  9,  2),
        (Color::Black,  8,  2), (Color::White,  7,  2), (Color::Black,  7,  3), (Color::White,  8,  1),
        (Color::Black,  8,  3), (Color::White,  6,  2), (Color::Black, 10,  5), (Color::White, 11,  5),
        (Color::Black, 11,  6), (Color::White, 12,  6), (Color::Black, 11,  7), (Color::White, 13,  5),
        (Color::Black, 12,  7), (Color::White,  5, 16), (Color::Black,  3, 14), (Color::White,  3, 17),
        (Color::Black,  2, 17), (Color::White,  4, 16), (Color::Black,  3, 15), (Color::White,  7, 15),
        (Color::Black,  4, 17), (Color::White,  5, 17), (Color::Black,  3, 18), (Color::White,  9,  7),
        (Color::Black,  8,  7), (Color::White,  8,  8), (Color::Black,  7,  7), (Color::White,  9,  8),
        (Color::Black,  9,  6), (Color::White,  7,  8), (Color::Black,  6,  7), (Color::White, 11,  9),
        (Color::Black,  6,  8), (Color::White,  6,  9), (Color::Black,  5,  9), (Color::White,  6, 10),
        (Color::Black,  6, 15), (Color::White,  6, 16), (Color::Black,  7, 14), (Color::White,  8, 15),
        (Color::Black,  6, 13), (Color::White,  5, 14), (Color::Black,  5, 13), (Color::White,  6, 14),
        (Color::Black,  4, 14), (Color::White,  5, 15), (Color::Black,  5, 10), (Color::White,  7, 12),
        (Color::Black,  6, 11), (Color::White,  7, 11), (Color::Black, 15, 11), (Color::White, 13, 10),
        (Color::Black,  7, 13), (Color::White,  8, 13), (Color::Black,  8, 14), (Color::White,  9, 14),
        (Color::Black,  8, 12), (Color::White,  9, 13), (Color::Black,  6, 12), (Color::White,  7, 10),
        (Color::Black,  2,  6), (Color::White,  3,  6), (Color::Black,  1,  5), (Color::White,  1,  4),
        (Color::Black,  3,  7), (Color::White,  2,  7), (Color::Black,  1,  6), (Color::White,  1,  7),
        (Color::Black,  4,  6), (Color::White,  3,  5), (Color::Black,  0,  4), (Color::White,  2,  4),
        (Color::Black,  2,  2), (Color::White,  1,  2), (Color::Black,  4,  5), (Color::White,  5,  3),
        (Color::Black,  4,  3), (Color::White,  3,  2), (Color::Black,  5,  4), (Color::White,  6,  3),
        (Color::Black,  6,  4), (Color::White,  1, 12), (Color::Black,  1, 13), (Color::White,  0, 13),
        (Color::Black,  0, 14), (Color::White,  0, 12), (Color::Black,  1, 14), (Color::White, 16,  6),
        (Color::Black, 17,  5), (Color::White, 15,  5), (Color::Black, 17,  6), (Color::White, 16,  7),
        (Color::Black, 17,  7), (Color::White, 16,  9), (Color::Black, 13,  6), (Color::White, 12,  5),
        (Color::Black, 12,  9), (Color::White, 12, 10), (Color::Black, 10,  9), (Color::White, 10,  8),
        (Color::Black, 11,  8), (Color::White, 11, 10), (Color::Black,  8, 10), (Color::White,  8, 11),
        (Color::Black,  9, 11), (Color::White,  9, 12), (Color::Black,  7,  9), (Color::White,  8,  9),
        (Color::Black,  2, 10), (Color::White,  1, 10), (Color::Black,  3, 10), (Color::White,  2,  9),
        (Color::Black,  3, 11), (Color::White,  2, 12), (Color::Black,  3, 12), (Color::White,  1,  9),
        (Color::Black,  3,  8), (Color::White,  3,  9), (Color::Black,  4,  9), (Color::White, 17, 11),
        (Color::Black, 15, 12), (Color::White, 17,  8), (Color::Black, 14, 13), (Color::White, 13, 13),
        (Color::Black, 14, 11), (Color::White, 13, 12), (Color::Black, 17, 12), (Color::White, 16, 11),
        (Color::Black, 15,  8), (Color::White, 16,  8), (Color::Black, 14,  6), (Color::White, 15,  6),
        (Color::Black,  9, 17), (Color::White,  8, 17), (Color::Black, 10, 17), (Color::White, 12, 16),
        (Color::Black, 12, 17), (Color::White, 14, 14), (Color::Black, 15, 13), (Color::White, 13,  7),
        (Color::Black, 14,  7), (Color::White, 13,  8), (Color::Black, 13,  9), (Color::White, 14,  9),
        (Color::Black, 12,  8), (Color::White, 14,  8), (Color::Black, 15,  9), (Color::White, 14, 10),
        (Color::Black, 15, 10), (Color::White, 15,  7), (Color::Black, 10,  6), (Color::White, 18, 11),
        (Color::Black, 13, 11), (Color::White, 12, 11), (Color::Black, 14,  0), (Color::White, 13,  0),
        (Color::Black, 15,  0), (Color::White, 13,  1), (Color::Black, 18, 12), (Color::White, 16,  4),
        (Color::Black, 17,  4), (Color::White,  9, 18), (Color::Black, 10, 18), (Color::White,  8, 18),
        (Color::Black, 16, 10), (Color::White, 17, 10), (Color::Black, 15, 14), (Color::White, 16, 12),
        (Color::Black, 17, 13), (Color::White, 18,  7), (Color::Black, 18,  6), (Color::White, 18,  8),
        (Color::Black,  5, 18), (Color::White,  6, 18), (Color::Black,  4, 18), (Color::White, 10, 16)
    ];

    // white should win by 48.5 points
    let (value, _policy) = predict(&moves, Color::Black);

    assert!(value < 0.0, "White should win by 48.5 -- {}", value);
}
