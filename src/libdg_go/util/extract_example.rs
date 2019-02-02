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

use color::Color;
use board::Board;
use ::DEFAULT_KOMI;

use super::features::{HWC, FEATURE_SIZE, NUM_FEATURES, Features};
use super::sgf::{Sgf, SgfError};
use super::symmetry;

use libc::{c_float, c_char, c_int, c_uchar};
use rand::prelude::SliceRandom;
use regex::Regex;
use std::ffi::CStr;

#[repr(C)]
pub struct Example {
    pub features: [c_float; FEATURE_SIZE],
    pub index: c_int,
    pub color: c_int,
    pub policy: [c_uchar; 905],
    pub winner: c_int,
    pub number: c_int
}

impl Default for Example {
    fn default() -> Example {
        Example {
            features: [0.0; FEATURE_SIZE],
            index: 0,
            color: 0,
            policy: [0; 905],
            winner: 0,
            number: 0
        }
    }
}

struct Candidate<'a> {
    board: Board,
    index: usize,
    color: Color,
    policy: Option<&'a [u8]>,
    value: Option<f32>
}

impl Candidate<'_> {
    /// Returns true if this candidate has an MCTS policy.
    fn has_policy(&self) -> bool {
        self.policy.is_some()
    }

    /// Returns true if this candidate has a _reasonable_ win rate, i.e. between `[-0.95, 0.95]`.
    fn has_reasonable_value(&self) -> bool {
        match self.value {
            None => true,
            Some(v) => v.abs() < 0.95
        }
    }
}

/// Returns the number of features used internally.
#[no_mangle]
pub unsafe extern fn get_num_features() -> c_int {
    NUM_FEATURES as i32
}

/// Extract a single example from the given SGF file. If the file contains
/// multiple examples, then a random one is picked.
///
/// # Arguments
///
/// - `raw_sgf_content` - The UTF-8 encoded content of an SGF file.
///
#[no_mangle]
pub unsafe extern fn extract_single_example(
    raw_sgf_content: *const c_char,
    out: *mut Example
) -> c_int
{
    lazy_static! {
        static ref EMPTY_POLICY: String = "0".repeat(905);

        static ref WINNER: Regex = Regex::new(r"RE\[([^\]]+)\]").unwrap();
        static ref SCORED: Regex = Regex::new(r"RE\[[BW]\+[0-9\.]+\]").unwrap();
        static ref KOMI: Regex = Regex::new(r"KM\[([^\]]*)\]").unwrap();
    }

    CStr::from_ptr(raw_sgf_content as *const _).to_str().map(|content| {
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
        let mut has_policy = false;
        let mut pass_count = 0;

        for m in Sgf::new(content.as_bytes(), komi) {
            match m {
                Err(SgfError::IllegalMove) => { return -30 },
                Err(SgfError::ParseError) => { return -23 },
                Ok(m) => {
                    let is_pass = m.x >= 19 || m.y >= 19;

                    pass_count = if is_pass { pass_count + 1 } else { 0 };
                    has_policy = has_policy || m.policy.is_some();
                    examples.push(Candidate {
                        board: m.board,
                        index: if is_pass { 361 } else { 19 * m.y + m.x },
                        color: m.color,
                        policy: m.policy,
                        value: m.value
                    });
                }
            }
        }

        // if the game was scored, then add two passing moves to the end of
        // the game. This is necessary since a lot of games seems to be
        // missing them.
        while SCORED.is_match(&content) && pass_count < 2 {
            let last_board = examples.last().map(|cand| cand.board.clone());
            let last_color = examples.last().map(|cand| cand.color).unwrap_or(Color::White);

            examples.push(Candidate {
                board: last_board.unwrap_or_else(|| Board::new(komi)),
                index: 361,
                color: last_color.opposite(),
                policy: None,
                value: None
            });
            pass_count += 1;
        }

        // do not output games that had a questionable number of moves (early
        // resignations, or huge early blunders)
        if examples.len() < 30 {
            return -31;
        }

        // if any of the candidate examples has full policies, then only consider
        // those policies. Also remove any candidates whose `value` is too extreme
        // since the MCTS does not tend to play too well in those situations.
        let candidate_examples: Vec<usize> = (0..examples.len())
            .filter(|&i| {
                (!has_policy || examples[i].has_policy()) && examples[i].has_reasonable_value()
            }).collect();

        candidate_examples.choose(&mut rand::thread_rng()).map(|&i| {
            let features = examples[i].board.get_features::<HWC, f32>(
                examples[i].color,
                symmetry::Transform::Identity
            );

            (*out).features.clone_from_slice(&features);
            (*out).index = examples[i].index as c_int;
            (*out).color = examples[i].color as c_int;
            (*out).policy.clone_from_slice(match examples[i].policy {
                Some(ref policy) => {
                    assert_eq!(policy.len(), 905, "illegal policy -- {:?}", policy);

                    policy
                },
                None => EMPTY_POLICY.as_bytes()
            });
            (*out).winner = winner as c_int;
            (*out).number = i as c_int;

            0
        }).unwrap_or(-30)
    }).unwrap_or(-1) as c_int
}

