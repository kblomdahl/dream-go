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

#[macro_use] extern crate lazy_static;
extern crate libc;
extern crate rand;
extern crate regex;

mod asm;
mod board;
#[macro_use] mod board_fast;
mod circular_buf;
mod codegen;
mod color;
mod features;
mod ladder;
mod small_set;
mod score;
pub mod sgf;
pub mod symmetry;
mod zobrist;

pub use self::color::*;
pub use self::board::*;
pub use self::features::*;
pub use self::ladder::*;
pub use self::score::*;

pub const DEFAULT_KOMI: f32 = 7.5;

/* -------- extract_single_example -------- */

#[repr(C)]
pub struct Example {
    features: [libc::c_char; FEATURE_SIZE],
    index: libc::c_int,
    color: libc::c_int,
    policy: [libc::c_uchar; 905],
    winner: libc::c_int,
    number: libc::c_int
}

struct Candidate {
    board: Board,
    index: usize,
    color: Color,
    policy: Option<String>
}

/// Extract a single example from the given SGF file. If the file contains
/// multiple examples, then a random one is picked.
///
/// # Arguments
///
/// - `raw_sgf_content` - The UTF-8 encoded content of an SGF file.
///
#[no_mangle]
pub extern fn extract_single_example(
    raw_sgf_content: *const libc::c_char,
    out: *mut Example
) -> libc::c_int
{
    use rand::Rng;
    use regex::Regex;
    use std::ffi::CStr;
    use sgf::SgfCoordinate;

    lazy_static! {
        static ref EMPTY_POLICY: String = "0".repeat(905);

        static ref WINNER: Regex = Regex::new(r"RE\[([^\]]+)\]").unwrap();
        static ref SCORED: Regex = Regex::new(r"RE\[[BW]\+[0-9\.]+\]").unwrap();
        static ref KOMI: Regex = Regex::new(r"KM\[([^\]]*)\]").unwrap();
        static ref MOVE: Regex = Regex::new(r";([BW])\[([^\]]*)\](?:P\[([^\]]*)\])?").unwrap();
    }

    unsafe { CStr::from_ptr(raw_sgf_content) }.to_str().map(|content| {
        // find the komi by looking for the pattern `KM[...]` at any point
        // in the file.
        let komi = {
            if let Some(caps) = KOMI.captures(&content) {
                if caps[1] == *"0" {
                    DEFAULT_KOMI  // foxwq always output an empty komi
                } else {
                    match caps[1].parse::<f32>() {
                        Ok(komi) => komi,
                        Err(_) => { return -21; },
                    }
                }
            } else {
                DEFAULT_KOMI
            }
        };

        // find the winner by looking for the pattern `KM[...]`.
        let winner = {
            if let Some(caps) = WINNER.captures(&content) {
                match caps[1].chars().nth(0) {
                    Some('B') => Color::Black,
                    Some('W') => Color::White,
                    _ => { return -22; }
                }
            } else {
                return -22;
            }
        };

        // find _all_ recorded moves, and their policies (if applicable).
        let mut examples = vec! [];
        let mut board = Board::new(komi);
        let mut pass_count = 0;

        for moves in MOVE.captures_iter(&content) {
            let (x, y) = sgf::CGoban::parse(&moves[2]).unwrap_or_else(|_err| { (19, 19) });
            let policy = moves.get(3).map(|input| input.as_str().to_string());
            let current_color = match &moves[1] {
                "B" => Color::Black,
                "W" => Color::White,
                _ => unreachable!()
            };

            if x >= 19 || y >= 19 {
                examples.push(Candidate {
                    board: board.clone(),
                    index: 361,
                    color: current_color,
                    policy: policy
                });
                pass_count += 1;
            } else if board.is_valid(current_color, x, y) {
                examples.push(Candidate {
                    board: board.clone(),
                    index: 19 * y + x,
                    color: current_color,
                    policy: policy
                });
                board.place(current_color, x, y);
                pass_count = 0;
            } else {
                return -20;
            }
        }

        // if the game was scored, then add two passing moves to the end of
        // the game. This is necessary since a lot of games seems to be
        // missing them.
        while SCORED.is_match(&content) && pass_count < 2 {
            let last_color = examples.last().map(|cand| cand.color).unwrap_or(Color::White);

            examples.push(Candidate {
                board: board.clone(),
                index: 361,
                color: last_color.opposite(),
                policy: None
            });
            pass_count += 1;
        }

        // do not output games that had a questionable number of moves (early
        // resignations, or huge early blunders)
        if examples.len() < 50 {
            return -31;
        }

        // if any of the candidate examples has full policies, then only consider
        // those policies.
        let candidate_examples: Vec<usize> = if examples.iter().any(|cand| cand.policy.is_some()) {
            (0..examples.len()).filter(|&i| examples[i].policy.is_some()).collect()
        } else {
            (0..examples.len()).collect()
        };

        rand::thread_rng().choose(&candidate_examples).map(|&i| {
            let features = examples[i].board.get_features::<CHW>(
                examples[i].color,
                symmetry::Transform::Identity
            );

            unsafe {
                (*out).features.clone_from_slice(&features);
                (*out).index = examples[i].index as libc::c_int;
                (*out).color = examples[i].color as libc::c_int;
                (*out).policy.clone_from_slice(match examples[i].policy {
                    Some(ref policy) => {
                        assert!(policy.len() == 905, "illegal policy -- {}", policy);

                        policy.as_bytes()
                    },
                    None => EMPTY_POLICY.as_bytes()
                });
                (*out).winner = winner as libc::c_int;
                (*out).number = i as libc::c_int;
            }

            0
        }).unwrap_or(-30)
    }).unwrap_or(-1) as libc::c_int
}
