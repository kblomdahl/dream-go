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
#[macro_use] extern crate lazy_static;
extern crate regex;
extern crate dream_go;
extern crate ordered_float;

mod common;

use common::{playout_file, predict, greedy_score};
use dream_go::go::{Color};
use ordered_float::OrderedFloat;

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

/// Test that the engine does not play `j13` as white in the
/// following board position since it is a broken ladder:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │                                       │ 19
/// 18 │               ○ ● ● ○ ● ● ● ●         │ 18
/// 17 │       ○       ○ ● ○ ○ ○ ○ ○ ○ ● ●     │ 17
/// 16 │       ○     ● ● ○ ●           ○ ○     │ 16
/// 15 │             ● ○ ○ ●                   │ 15
/// 14 │       ● ● ● ○ ● ○ ●                   │ 14
/// 13 │       ● ○ ○ ○ ●                       │ 13
/// 12 │     ○ ○ ● ○   ●                       │ 12
/// 11 │         ● ○                           │ 11
/// 10 │       ○ ● ○                           │ 10
///  9 │   ○     ● ○                           │ 9
///  8 │   ● ○   ● ○                           │ 8
///  7 │   ● ○                                 │ 7
///  6 │     ● ○ ○                       ●     │ 6
///  5 │     ● ●                               │ 5
///  4 │                               ●       │ 4
///  3 │       ●                               │ 3
///  2 │                                       │ 2
///  1 │                                       │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
/// 
#[test]
fn ladder_1() {
    let board = playout_file("examples/ladder_1.sgf", Some(70));
    let (_value, policy) = predict(&board, Color::White);
    let (x, y) = policy_to_vertex(&policy);

    assert!(x != 8 || y != 6, "Broken ladder at (8, 6) -- ({}, {})", x, y);
}

/// Test that the engine does not play `f5` as black in the
/// following board position since it is a broken ladder:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │                                       │ 19
/// 18 │                                       │ 18
/// 17 │                                       │ 17
/// 16 │       ○                       ●       │ 16
/// 15 │                                       │ 15
/// 14 │                                       │ 14
/// 13 │                                       │ 13
/// 12 │                                       │ 12
/// 11 │                                       │ 11
/// 10 │                                       │ 10
///  9 │                                       │ 9
///  8 │                                       │ 8
///  7 │                               ●       │ 7
///  6 │                                       │ 6
///  5 │             ○                         │ 5
///  4 │       ○ ○ ● ● ○               ●       │ 4
///  3 │       ○ ● ○ ○ ●                       │ 3
///  2 │       ● ●                             │ 2
///  1 │                                       │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
/// 
#[test]
fn ladder_2() {
    let board = playout_file("examples/ladder_2.sgf", Some(18));
    let (_value, policy) = predict(&board, Color::Black);
    let (x, y) = policy_to_vertex(&policy);

    assert!(x != 5 || y != 14, "Broken ladder at (5, 14) -- ({}, {})", x, y);
}

/// That that the engine does not play `q10` as white in the
/// following board position since it is a broken ladder:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │                                       │ 19
/// 18 │                       ○   ○ ●         │ 18
/// 17 │           ○   ●               ● ○ ●   │ 17
/// 16 │       ○           ●   ●   ○   ● ○ ●   │ 16
/// 15 │                             ● ● ● ○   │ 15
/// 14 │     ○             ○   ○   ○   ○ ○   ○ │ 14
/// 13 │                                   ○   │ 13
/// 12 │                       ●   ●   ● ● ○   │ 12
/// 11 │                             ● ○ ○ ●   │ 11
/// 10 │                                 ●     │ 10
///  9 │                                       │ 9
///  8 │                                       │ 8
///  7 │   ●                                   │ 7
///  6 │ ●   ●                                 │ 6
///  5 │   ● ○ ○                               │ 5
///  4 │   ○ ● ○                   ●   ●       │ 4
///  3 │   ○ ● ○                               │ 3
///  2 │                                       │ 2
///  1 │                                       │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
/// 
#[test]
fn ladder_3() {
    let board = playout_file("examples/ladder_3.sgf", Some(54));
    let (_value, policy) = predict(&board, Color::White);
    let (x, y) = policy_to_vertex(&policy);

    assert!(x != 15 || y != 9, "Broken ladder at (15, 9) -- ({}, {})", x, y);
}

/// Test that the engine correctly detects that a large white dragon is dead and
/// black should win the game by 38.5 points:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │       ○ ○ ●         ● ○ ○             │ 19
/// 18 │     ○ ○ ● ●       ● ● ● ○ ○ ○         │ 18
/// 17 │     ○ ● ● ○   ● ●     ● ● ○ ● ○ ○ ○ ○ │ 17
/// 16 │     ○ ●   ●   ● ○ ●     ● ● ● ● ● ○   │ 16
/// 15 │     ○ ●       ● ○ ● ● ● ● ○ ○ ● ● ● ○ │ 15
/// 14 │     ○ ○ ● ●   ● ○ ○ ○ ● ○ ○ ● ● ○ ○   │ 14
/// 13 │     ○ ● ○       ● ● ○ ○ ○ ● ○ ○ ○     │ 13
/// 12 │     ○ ● ● ● ●     ● ○ ● ● ● ○ ○   ●   │ 12
/// 11 │   ○ ● ● ○ ○ ● ●   ● ○ ●   ● ○     ●   │ 11
/// 10 │     ○ ○ ○   ○ ○ ● ● ○ ● ● ● ○ ○ ○ ○ ○ │ 10
///  9 │   ○   ● ● ○   ○ ● ○ ○ ○ ○ ● ● ● ○     │ 9
///  8 │   ○ ● ● ○ ○ ●   ● ○ ● ● ○   ●   ● ○ ○ │ 8
///  7 │   ○ ● ○   ○ ● ● ● ○   ○ ○ ○ ● ● ● ○   │ 7
///  6 │ ● ● ● ○   ●   ● ○ ○ ○   ● ○ ○ ●   ● ○ │ 6
///  5 │ ● ● ●   ○ ○ ● ● ● ○   ● ● ● ●     ● ○ │ 5
///  4 │ ○ ● ○ ○   ● ● ○ ○ ○ ○ ● ○ ● ●   ● ● ○ │ 4
///  3 │ ○ ○ ○   ○ ○ ○ ● ● ● ● ○   ●   ● ○ ○   │ 3
///  2 │         ○ ● ● ●     ● ○ ●   ● ● ○     │ 2
///  1 │         ○ ○ ○ ●     ● ●     ● ○ ○     │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
///
#[test]
fn dead_dragon_1() {
    let board = playout_file("examples/dead_dragon_1.sgf", None);
    let (value, _policy) = predict(&board, Color::Black);
    let (black, white) = greedy_score(&board, Color::Black);

    assert!(black > white, "Black should win by 31 (without komi) -- black {}, white {}", black, white);
    assert!(value > 0.0, "Black should win by 38.5 -- {}", value);
}

/// Test that the engine correctly detects that a large white dragon is dead and
/// black should win the game by 4.5 points:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │             ● ● ○ ○ ○ ○ ○     ○   ○   │ 19
/// 18 │       ● ● ●   ○ ● ○ ● ○ ○ ○ ○ ○ ○ ● ● │ 18
/// 17 │     ● ● ○ ● ○   ● ○ ● ● ● ● ● ○ ●   ● │ 17
/// 16 │ ● ● ● ○ ○ ○ ● ●   ● ●   ● ○ ○ ○ ● ● ● │ 16
/// 15 │ ○ ● ○   ○ ○ ○ ○ ● ●     ● ○ ○ ●   ●   │ 15
/// 14 │ ○ ○ ○   ● ○ ● ○ ○ ● ●     ● ● ●       │ 14
/// 13 │         ● ●   ○   ○ ● ● ● ●       ● ● │ 13
/// 12 │                   ○ ● ○ ○   ● ●   ●   │ 12
/// 11 │     ○           ○ ○ ● ● ○ ○ ○ ● ● ● ○ │ 11
/// 10 │   ● ○ ○   ○ ○ ○ ● ○ ○ ● ○     ○ ○ ○   │ 10
///  9 │   ○     ○     ○ ● ● ● ● ● ○ ○ ● ●     │ 9
///  8 │     ○   ○ ○     ○   ● ○ ○ ○   ○ ● ○   │ 8
///  7 │ ○   ○       ○ ○ ○ ○ ○ ● ● ○   ○ ○     │ 7
///  6 │ ● ○ ○ ○ ○ ○ ● ○ ● ● ● ●   ● ○         │ 6
///  5 │ ● ● ● ○ ●   ● ○ ●     ● ● ● ○ ● ●     │ 5
///  4 │     ● ● ● ● ● ● ● ●   ● ○ ○ ○ ○ ●     │ 4
///  3 │ ● ●   ● ○ ● ○ ○ ○ ● ● ○   ○   ○ ○ ○   │ 3
///  2 │   ● ○ ○ ○ ○ ○ ● ● ● ○ ○ ○     ○       │ 2
///  1 │ ● ○ ○ ○     ○ ○ ● ● ○   ● ○   ○       │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
/// 
#[test]
fn dead_dragon_2() {
    let board = playout_file("examples/dead_dragon_2.sgf", None);
    let (value, _policy) = predict(&board, Color::Black);
    let (black, white) = greedy_score(&board, Color::Black);

    assert!(black + 8 > white , "Black should win by 4.5 -- black {}, white {}", black, white);
    assert!(value > 0.0, "Black should win by 4.5 -- {}", value);
}

/// Test that the engine correctly detects that some black dragon is dead and
/// white should win the game by 7.5 points:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │   ●   ○ ● ○ ○     ○ ○ ●               │ 19
/// 18 │       ● ● ● ○     ○ ● ● ●             │ 18
/// 17 │ ● ● ● ● ○ ○ ○   ○ ○ ● ○ ● ● ● ● ● ●   │ 17
/// 16 │ ○ ● ○ ○   ○ ●     ● ○ ○ ○ ○ ○ ○ ○ ● ● │ 16
/// 15 │ ○ ○     ○ ○ ● ● ● ● ● ●     ○ ○ ○ ○ ● │ 15
/// 14 │ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ● ● ● ● ○ ● ○ ○ │ 14
/// 13 │ ● ● ●   ○ ○ ● ● ○ ○ ○ ○ ● ○ ● ● ● ○   │ 13
/// 12 │ ○   ● ○   ○ ●   ● ● ○ ○ ●   ● ○ ○     │ 12
/// 11 │   ● ● ○   ○ ● ●   ● ○ ● ● ●   ● ○ ○ ○ │ 11
/// 10 │   ● ○ ○ ○ ○ ●   ● ○ ● ● ● ○ ● ● ○ ●   │ 10
///  9 │   ○ ● ○ ● ● ● ● ○ ○ ○ ○ ○ ○   ● ● ○   │ 9
///  8 │   ● ● ● ● ○ ○   ○     ○   ○   ● ○   ○ │ 8
///  7 │         ● ● ○ ○ ○ ● ● ○   ○ ● ● ○ ○   │ 7
///  6 │           ○ ● ● ● ● ○ ○   ○ ○ ○   ○   │ 6
///  5 │     ●     ○ ●       ● ○   ●   ○ ○ ●   │ 5
///  4 │         ● ○ ● ○     ● ○ ○     ○ ●   ● │ 4
///  3 │       ●   ● ○       ● ● ○ ○ ○ ● ● ●   │ 3
///  2 │           ●         ● ○ ● ○ ●   ●     │ 2
///  1 │                         ● ●           │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
/// 
#[test]
fn dead_dragon_3() {
    let board = playout_file("examples/dead_dragon_3.sgf", None);
    let (value, _policy) = predict(&board, Color::White);
    let (black, white) = greedy_score(&board, Color::White);

    assert!(white + 8 > black, "White should win by 7.5 -- black {}, white {}", black, white);
    assert!(value > 0.0, "White should win by 7.5 -- {}", value);
}

/// Test that the engine correctly detects that some white dragon is dead and
/// black should win the game by 140.5 points:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │       ● ●   ● ● ○ ○ ○ ○ ● ● ○ ○   ○   │ 19
/// 18 │ ● ●     ●   ○   ● ○   ●     ● ○ ○   ○ │ 18
/// 17 │         ● ● ● ● ● ○ ○ ● ●   ● ● ● ○   │ 17
/// 16 │ ● ● ● ● ● ○ ● ● ○ ○ ● ○ ○ ● ● ● ○ ○   │ 16
/// 15 │ ● ○ ● ● ○ ○ ○ ● ● ○ ●     ○     ● ○   │ 15
/// 14 │ ○ ○ ● ● ● ○ ○ ○ ○ ○ ●         ● ● ○   │ 14
/// 13 │ ● ○ ○ ○ ○ ○   ○ ● ○ ● ● ●   ●   ● ○   │ 13
/// 12 │ ● ● ● ○   ○ ● ○ ● ● ○ ○ ● ● ● ○ ○ ○   │ 12
/// 11 │ ○ ● ● ● ● ○ ● ● ● ○ ○ ○ ○ ● ○ ○ ● ○ ○ │ 11
/// 10 │ ○ ● ● ○ ● ● ●       ● ● ○ ○ ○ ○ ● ● ○ │ 10
///  9 │ ○ ○ ● ○ ● ○ ○ ● ● ● ● ○ ○ ○ ● ● ● ● ● │ 9
///  8 │   ○ ● ○ ○   ○ ●     ● ● ○ ○ ○ ○ ○ ● ● │ 8
///  7 │   ○ ○ ○   ○ ○ ○ ●   ●   ○ ● ● ○ ●     │ 7
///  6 │     ○ ○     ○ ● ●   ●   ○ ● ● ● ● ● ● │ 6
///  5 │         ○ ○ ● ●     ●   ○ ● ●     ●   │ 5
///  4 │     ○ ○ ● ● ●       ●   ○ ●           │ 4
///  3 │       ○ ○ ● ●     ● ● ● ● ○ ● ● ● ● ● │ 3
///  2 │     ○ ○ ●   ●   ● ○ ●   ○ ○ ●     ● ○ │ 2
///  1 │     ○ ● ● ● ●   ●             ● ○ ○   │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
#[test]
fn dead_dragon_4() {
    let board = playout_file("examples/dead_dragon_4.sgf", None);
    let (value, _policy) = predict(&board, Color::Black);
    let (black, white) = greedy_score(&board, Color::Black);

    assert!(black > white, "Black should win by 140.5 -- black {}, white {}", black, white);
    assert!(value > 0.0, "Black should win by 140.5 -- {}", value);
}

/// Test that the engine correctly detects that some groups are in seki and that
/// white should win by 4.5:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │ ○ ○   ○ ○ ○ ○   ○ ○ ○     ○ ○ ● ●     │ 19
/// 18 │       ○ ○   ○   ○     ○   ○   ○ ● ●   │ 18
/// 17 │ ○           ○ ○ ○         ○   ○ ○ ●   │ 17
/// 16 │ ○ ○   ○       ○ ○ ○ ○     ○   ○ ● ●   │ 16
/// 15 │ ○     ○     ○ ○   ○     ○ ● ● ○ ○ ●   │ 15
/// 14 │ ○ ○ ○ ○ ○ ○ ● ● ○ ○   ○ ○ ○ ● ○ ●     │ 14
/// 13 │   ○ ○ ● ○ ● ● ○ ○ ○     ○ ○ ○ ● ●     │ 13
/// 12 │ ○ ○ ● ● ● ● ● ● ○ ○   ○ ○ ○ ●   ● ● ● │ 12
/// 11 │   ○ ○ ●   ○ ● ○ ● ○ ○ ○ ● ○ ●   ●     │ 11
/// 10 │ ○ ○ ○ ● ● ○ ● ○ ● ● ○ ● ● ○ ●   ●   ● │ 10
///  9 │ ● ○ ● ●   ○ ● ○ ○ ● ● ● ● ○ ● ●     ● │ 9
///  8 │ ● ● ● ○ ○ ○ ○ ○ ● ● ● ● ○ ○ ○ ● ●     │ 8
///  7 │ ● ○ ● ● ○ ○ ● ●   ● ●   ● ● ● ● ●     │ 7
///  6 │ ● ○ ○ ● ○ ● ● ●   ● ● ○       ●   ●   │ 6
///  5 │ ○ ○ ○ ● ● ○   ●     ●   ● ●   ●   ● ● │ 5
///  4 │ ○     ○ ○ ○ ○ ●       ●   ●     ● ● ● │ 4
///  3 │ ○   ○ ○ ○ ● ● ● ●     ●   ● ●       ● │ 3
///  2 │ ○   ○ ○   ○ ● ● ●   ●   ●   ●   ●   ● │ 2
///  1 │   ○ ○   ○ ○ ○ ○ ●   ● ●     ● ●   ●   │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
#[test]
fn seki_1() {
    let board = playout_file("examples/seki_1.sgf", None);
    let (value, _policy) = predict(&board, Color::White);
    let (black, white) = greedy_score(&board, Color::White);

    assert!(white + 8 > black, "White should win by 4.5 -- black {}, white {}", black, white);
    assert!(value > 0.0, "White should win by 4.5 -- {}", value);
}

/// Test that the engine correctly detects that some groups are in seki and that
/// white should win by 7.5:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │     ○   ○ ○ ○ ○ ○ ● ● ● ○ ○ ○ ○ ●     │ 19
/// 18 │     ○   ○   ○ ○ ● ● ○ ○ ○     ○ ●   ● │ 18
/// 17 │ ○ ○ ○ ○ ● ○ ○ ○ ○ ● ● ○ ○   ○ ○ ●     │ 17
/// 16 │ ● ○ ● ● ● ○ ● ● ○ ● ● ● ○ ○ ● ○ ● ● ● │ 16
/// 15 │ ● ● ●     ●   ● ○ ○ ○ ● ● ● ● ○ ● ● ○ │ 15
/// 14 │     ○   ●   ● ● ○ ○ ●   ● ○ ○ ○ ○ ○ ○ │ 14
/// 13 │               ○ ● ○ ● ●   ● ○ ○ ○ ○   │ 13
/// 12 │   ● ●           ● ●   ● ● ● ● ○ ○   ○ │ 12
/// 11 │ ●     ●     ● ● ●   ● ● ● ○ ● ● ○     │ 11
/// 10 │ ●   ● ●   ● ○ ● ○ ●   ● ○ ○ ● ○   ○   │ 10
///  9 │       ● ● ● ○   ○ ● ● ●   ○ ● ○     ○ │ 9
///  8 │ ● ● ● ○ ● ○ ○ ○ ○ ○ ○ ● ○ ○ ○ ○ ○ ○   │ 8
///  7 │ ● ○ ● ○ ● ○         ○ ○ ● ●     ○ ○ ○ │ 7
///  6 │ ● ○ ○ ○ ○ ○ ○   ○       ○ ○ ○ ○ ● ○ ● │ 6
///  5 │ ● ● ● ○   ○ ● ○ ○ ○ ○       ○ ○ ● ● ● │ 5
///  4 │ ●   ●   ○ ○ ● ● ○ ○ ○ ○ ○ ○   ○ ●   ○ │ 4
///  3 │ ●   ● ● ● ●   ● ● ● ○         ○ ● ○   │ 3
///  2 │ ○ ●   ●     ● ○ ● ● ○     ○ ○ ● ● ● ○ │ 2
///  1 │   ○ ● ●         ● ○ ○     ○ ● ●   ○   │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
#[test]
fn seki_2() {
    let board = playout_file("examples/seki_2.sgf", None);
    let (value, _policy) = predict(&board, Color::White);
    let (black, white) = greedy_score(&board, Color::White);

    assert!(white + 8 > black, "White should win by 7.5 -- black {}, white {}", black, white);
    assert!(value > 0.0, "White should win by 7.5 -- {}", value);
}

/// Test that the engine correctly detects that some groups are in seki and that
/// black should win by 19.5:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │ ○ ○ ○ ○   ○ ●   ○ ○ ○ ●   ○ ● ● ● ● ● │ 19
/// 18 │ ○ ○ ● ○ ○   ○ ○   ○ ● ● ● ●   ● ● ○ ● │ 18
/// 17 │ ○ ● ● ● ● ○ ○ ○ ○ ● ● ○ ● ● ● ● ○ ○ ○ │ 17
/// 16 │ ● ● ● ● ● ● ● ○ ● ○ ● ○ ● ○ ● ● ● ● ○ │ 16
/// 15 │     ●       ● ● ● ○ ○ ○ ○ ○ ● ● ○ ○ ○ │ 15
/// 14 │   ●     ● ● ●   ● ○ ● ○ ○   ○ ● ○ ○   │ 14
/// 13 │   ● ●     ● ● ● ● ● ● ● ○ ○ ○ ○ ○ ○ ○ │ 13
/// 12 │ ● ● ●               ● ○ ○ ● ● ● ○ ○   │ 12
/// 11 │           ● ● ●   ● ● ○ ● ● ● ○ ○     │ 11
/// 10 │ ● ●   ● ● ● ● ●   ● ● ●   ● ● ○ ● ● ● │ 10
///  9 │ ○ ● ● ● ○ ○ ● ○ ●         ● ● ○ ○ ○ ○ │ 9
///  8 │ ○   ● ○ ○ ○ ○ ○ ● ●   ● ● ● ● ○ ● ● ○ │ 8
///  7 │ ● ● ● ● ● ○ ○ ○ ● ○ ●   ● ○ ○ ● ●   ● │ 7
///  6 │ ○ ● ○   ○   ○ ● ●   ● ● ● ○   ○ ○ ●   │ 6
///  5 │ ○ ○ ○ ○ ○ ○ ○ ● ●   ● ○ ○ ○ ○   ○ ○ ○ │ 5
///  4 │ ○     ○ ● ○ ● ● ● ○ ● ○   ○ ○ ○ ● ● ● │ 4
///  3 │ ○ ○ ○ ● ● ● ● ● ● ● ○ ○ ○   ○ ● ● ●   │ 3
///  2 │     ○ ● ● ● ○ ● ○ ○ ○     ○ ● ●   ○ ● │ 2
///  1 │   ○ ● ● ●   ○ ○ ○ ○       ○ ○ ● ● ○   │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
#[test]
fn seki_3() {
    let board = playout_file("examples/seki_3.sgf", None);
    let (value, _policy) = predict(&board, Color::Black);
    let (black, white) = greedy_score(&board, Color::Black);

    assert!(black > white, "Black should win by 19.5 -- black {}, white {}", black, white);
    assert!(value > 0.0, "Black should win by 19.5 -- {}", value);
}

/// Test that the engine correctly detects that some groups are in seki and that
/// black should win by 7.5:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │ ●   ●   ●   ● ● ○   ○   ○   ○ ○ ○ ●   │ 19
/// 18 │ ● ●         ● ○ ○ ○   ○   ○   ○ ○ ○ ● │ 18
/// 17 │ ●   ● ● ●   ● ○ ○ ○ ○ ● ○ ○       ● ○ │ 17
/// 16 │   ●   ● ●   ● ● ● ○ ● ●   ○   ○ ○   ○ │ 16
/// 15 │       ●   ● ● ● ○ ○ ● ● ● ● ○ ○ ○ ○ ○ │ 15
/// 14 │       ●   ●   ● ● ●     ○ ● ○   ● ● ○ │ 14
/// 13 │ ○     ●           ● ● ● ● ○ ○ ● ●   ● │ 13
/// 12 │ ○   ● ●     ● ●   ● ● ○ ○ ○ ● ●   ● ● │ 12
/// 11 │ ● ● ●       ● ● ● ● ○     ○ ○ ●       │ 11
/// 10 │ ○ ● ●     ● ○ ○ ● ● ○       ○ ○ ● ● ● │ 10
///  9 │ ○ ○ ● ● ● ● ○ ● ○ ● ○           ○ ●   │ 9
///  8 │     ○   ● ● ○   ○ ○ ○ ○ ○   ● ○   ○ ○ │ 8
///  7 │ ○ ○ ○ ○ ● ○   ○ ○ ○ ● ● ○   ● ○       │ 7
///  6 │       ○ ○ ○ ○ ● ○ ● ● ○ ○   ●   ○ ○ ○ │ 6
///  5 │ ○   ○ ○ ● ○ ● ● ● ● ● ● ○ ○ ● ○ ○ ● ○ │ 5
///  4 │ ○ ○ ○ ● ● ● ● ● ● ● ○ ● ● ○ ○ ○ ● ● ● │ 4
///  3 │ ○ ● ○ ●   ●   ● ○ ○ ○ ○ ● ● ○ ○ ● ●   │ 3
///  2 │ ● ● ●   ●   ● ○   ○   ○ ● ● ○ ● ● ○ ○ │ 2
///  1 │           ● ○ ○ ○ ○ ○ ● ● ● ○ ●   ●   │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
#[test]
fn seki_4() {
    let board = playout_file("examples/seki_4.sgf", None);
    let (value, _policy) = predict(&board, Color::Black);
    let (black, white) = greedy_score(&board, Color::Black);

    assert!(black > white + 8, "Black should win by 7.5 -- black {}, white {}", black, white);
    assert!(value > 0.0, "Black should win by 7.5 -- {}", value);
}

/// Test that the engine correctly detects that the _bent four in the corner_ is
/// dead and therefore white wins:
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │   ● ○   ○ ○ ○ ○           ○ ● ● ●   ● │ 19
/// 18 │ ● ○ ○   ○ ○ ● ○ ○ ○ ○ ○ ○ ○ ○ ●   ●   │ 18
/// 17 │   ○     ○ ● ● ● ● ● ● ○ ○ ○ ○ ○ ●   ● │ 17
/// 16 │ ○   ○   ○ ○ ● ●       ● ○ ● ○ ● ● ● ● │ 16
/// 15 │ ○   ○   ○ ○ ● ○       ● ● ● ● ○ ○ ○ ● │ 15
/// 14 │   ○ ○ ○ ● ●   ● ●       ●   ● ○ ● ● ● │ 14
/// 13 │   ○ ○ ● ●       ●     ● ○ ○ ○ ○ ○ ● ○ │ 13
/// 12 │   ○ ●   ●   ● ● ●   ● ● ● ○     ○ ○ ○ │ 12
/// 11 │   ○ ● ● ●   ● ○   ● ● ○ ○ ○ ○   ○   ○ │ 11
/// 10 │   ○ ○ ● ●     ○ ●     ● ● ○ ○     ○   │ 10
///  9 │ ○ ○ ●   ●   ●   ●       ● ● ○   ○   ○ │ 9
///  8 │ ○ ● ●   ● ● ● ●   ●   ●   ● ○   ○ ○ ● │ 8
///  7 │ ● ○   ● ● ●   ● ● ● ●   ● ● ○   ○ ● ● │ 7
///  6 │ ● ● ●   ●     ● ○ ○ ● ● ● ● ● ○ ○ ○ ● │ 6
///  5 │     ●     ● ● ● ○ ○ ● ● ○ ○ ○ ● ● ● ● │ 5
///  4 │   ● ●   ● ● ○ ○   ○ ○ ○ ○   ○ ○ ○ ● ● │ 4
///  3 │ ● ●     ● ○ ○   ○   ○ ○ ○   ○ ○ ● ●   │ 3
///  2 │       ● ● ● ○ ○   ○ ○ ○     ○ ● ● ● ○ │ 2
///  1 │ ● ●     ● ● ● ○     ○ ○ ○   ○ ●   ○ ○ │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
#[test]
fn bent_four_1() {
    let board = playout_file("examples/bent_four_1.sgf", None);
    let (value, _policy) = predict(&board, Color::White);
    let (black, white) = greedy_score(&board, Color::White);

    assert!(white + 8 > black, "White should win by 6.5 -- black {}, white {}", black, white);
    assert!(value > 0.0, "White should win by 6.5 -- {}", value);
}

/// Test that the engine correctly detects that white has won the game (nothing
/// really special about the board position):
/// 
/// ```
/// 	 a b c d e f g h j k l m n o p q r s t
///    ╭───────────────────────────────────────╮
/// 19 │                           ○ ● ●       │ 19
/// 18 │                 ○         ○ ○ ● ●     │ 18
/// 17 │   ○ ● ○ ○   ○ ○ ● ○ ○     ● ○ ○ ●     │ 17
/// 16 │   ● ○ ○ ● ○ ○ ● ● ● ○         ○ ●     │ 16
/// 15 │ ● ○ ○ ○ ● ● ●     ● ○         ○ ○ ●   │ 15
/// 14 │   ● ○ ○ ●         ● ● ○ ○ ○   ○ ● ●   │ 14
/// 13 │   ● ● ○ ●         ● ● ● ○ ● ● ○ ○ ● ● │ 13
/// 12 │   ○ ○ ●     ● ● ● ○   ● ● ○ ● ○ ○ ● ○ │ 12
/// 11 │     ○ ●     ● ○ ○ ○ ○ ● ● ○ ○ ● ○ ○ ○ │ 11
/// 10 │   ○ ○ ○ ● ● ○   ○   ● ○ ● ● ○ ● ○     │ 10
///  9 │   ○ ● ●   ● ○ ○ ●     ○ ○ ○ ○ ● ● ○   │ 9
///  8 │     ○ ●     ● ○ ○ ●     ○ ● ● ● ○ ○ ○ │ 8
///  7 │ ○ ○ ○ ●     ● ○   ○       ○ ○ ● ○ ● ● │ 7
///  6 │ ○ ● ●     ● ● ● ○ ○       ○ ● ● ● ●   │ 6
///  5 │ ● ●   ● ● ○ ○ ● ● ○         ○ ●       │ 5
///  4 │       ●   ○   ○ ○   ○ ○   ○ ○ ○ ●     │ 4
///  3 │       ● ○ ○ ○ ●   ○ ○ ● ○ ○ ● ● ●     │ 3
///  2 │     ●   ● ○     ○ ● ● ● ● ●           │ 2
///  1 │       ● ● ● ○   ○ ○ ●             ●   │ 1
///    ╰───────────────────────────────────────╯
///      a b c d e f g h j k l m n o p q r s t
///     ● Black    ○ White
/// ```
#[test]
fn end_1() {
    let board = playout_file("examples/end_1.sgf", None);
    let (value, _policy) = predict(&board, Color::Black);
    let (black, white) = greedy_score(&board, Color::Black);

    assert!(white > black, "White should win by 41 (without komi) -- black {}, white {}", black, white);
    assert!(value < 0.0, "White should win by 48.5 -- {}", value);
}
