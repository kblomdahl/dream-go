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
use point::Point;
use ::DEFAULT_KOMI;

use super::features::{HWC, FEATURE_SIZE, NUM_FEATURES, Features};
use super::sgf::{Sgf, SgfEntry, SgfError};
use super::symmetry;

use dg_utils::types::f16;
use dg_utils::b85;
use utils::sgf::{CGoban, SgfCoordinate};

use libc::{c_char, c_int};
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{SeedableRng, Rng};
use regex::{Regex, Captures};
use std::ffi::CStr;
use std::sync::Mutex;
use ordered_float::OrderedFloat;

#[repr(C)]
pub struct Example {
    pub index: c_int,
    pub next_index: c_int,
    pub color: c_int,
    pub policy: [f32; 362],
    pub next_policy: [f32; 362],
    pub ownership: [f32; 361],
    pub winner: c_int,
    pub number: c_int,
    pub komi: f32,
    pub features: [f16; FEATURE_SIZE],
}

impl Default for Example {
    fn default() -> Example {
        Example {
            index: 0,
            next_index: 0,
            color: 0,
            policy: [f32::from(0.0); 362],
            next_policy: [f32::from(0.0); 362],
            ownership: [f32::from(0.0); 361],
            winner: 0,
            number: 0,
            komi: DEFAULT_KOMI,
            features: [f16::from(0.0); FEATURE_SIZE],
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
}

impl<'a> From<SgfEntry<'a>> for Candidate<'a> {
    fn from(m: SgfEntry<'a>) -> Self {
        Self {
            board: m.board,
            index: m.point.to_packed_index(),
            color: m.color,
            policy: m.policy,
            value: m.value
        }
    }
}

lazy_static! {
    static ref RNG: Mutex<StdRng> = Mutex::new(StdRng::from_entropy());
}

/// Returns the number of features used internally.
#[no_mangle]
pub unsafe extern fn get_num_features() -> c_int {
    NUM_FEATURES as i32
}

/// Sets the random seed used to determine which example is extracted from
/// each SGF file.
///
/// # Arguments
///
/// * `seed` -
///
#[no_mangle]
pub unsafe extern fn set_seed(seed: i32) {
    let mut rng = RNG.lock().unwrap();

    *rng = StdRng::seed_from_u64(seed as u64);
}

/// Extract a single example from the given SGF file. If the file contains
/// multiple examples, then a random one is picked.
///
/// # Arguments
///
/// - `raw_sgf_content` - The UTF-8 encoded content of an SGF file.
/// - `out` - Output of the extracted example.
///
#[no_mangle]
pub unsafe extern fn extract_single_example(
    raw_sgf_content: *const c_char,
    out: *mut Example
) -> c_int
{
    CStr::from_ptr(raw_sgf_content as *const _).to_str().map(|content| {
        let komi =
            match get_komi_from_sgf(content) {
                Ok(km) => km,
                Err(code) => { return code; }
            };

        // find _all_ recorded moves, and their policies (if applicable).
        let mut examples: Vec<Candidate> = Vec::with_capacity(254);
        let mut has_policy = false;
        let mut pass_count = 0;

        for m in Sgf::new(content.as_bytes(), komi) {
            match m {
                Err(SgfError::IllegalMove) => { return -30 },
                Err(SgfError::ParseError) => { return -23 },
                Ok(m) => {
                    let is_pass = m.point == Point::default();

                    pass_count = if is_pass { pass_count + 1 } else { 0 };
                    has_policy = has_policy || m.policy.is_some();
                    examples.push(Candidate::from(m));
                }
            }
        }

        // if the game was scored, then add two passing moves to the end of
        // the game. This is necessary since a lot of games seems to be
        // missing them.
        while is_scored(content) && pass_count < 2 {
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

        choose_example(&examples, has_policy).map(|i| {
            copy_candidates_to(content, &examples, i, &mut *out)
        }).unwrap_or(-32)
    }).unwrap_or(-1) as c_int
}

/// Choose a single example from the given examples. If there are given policies
/// among the examples then those are favoured.
/// 
/// # Arguments
/// 
/// * `examples` -
/// * `has_policy` - 
/// 
fn choose_example(examples: &[Candidate], has_policy: bool) -> Option<usize> {
    let candidate_examples: Vec<usize> = (0..examples.len())
        .filter(|&i| !has_policy || examples[i].has_policy())
        .collect();

    if candidate_examples.is_empty() {
        return None;
    }

    // choose a single example based on the value distribution, so that moves
    // closer to `0.5` are more likely to be picked.
    let mut cum_examples: Vec<OrderedFloat<f32>> = vec! [];
    let mut so_far: f32 = 0.0;

    for &i in &candidate_examples {
        let value =
            match examples[i].value {
                Some(val) => 0.5 - (val - 0.5).abs(),
                None => 0.5,
            };

        so_far += value;
        cum_examples.push(OrderedFloat(so_far));
    }

    let selected = RNG.lock().unwrap().sample(Uniform::new(0.0, so_far));

    match cum_examples.binary_search(&OrderedFloat(selected)) {
        Ok(i) => Some(i),
        Err(i) => Some(i)
    }
}

/// Set the given example values according to the given SGF and chosen examples.
/// 
/// # Arguments
/// 
/// * `content` - 
/// * `examples` - 
/// * `i` - 
/// * `out` - 
/// 
fn copy_candidates_to(
    content: &str,
    examples: &[Candidate],
    i: usize,
    out: &mut Example
) -> c_int
{
    const EMPTY_POLICY: [f32; 362] = [0.0; 362];

    let winner =
        match get_winner_from_sgf(content) {
            Ok(re) => re,
            Err(code) => { return code; }
        };
    let next_example = examples.get(i+1);
    let features = examples[i].board.get_features::<HWC, f16>(
        examples[i].color,
        symmetry::Transform::Identity
    );

    out.features.clone_from_slice(&features);
    out.index = examples[i].index as c_int;
    out.next_index = next_example.map(|example| example.index).unwrap_or(361) as c_int;
    out.color = examples[i].color as c_int;
    out.ownership.clone_from_slice(&get_vertex_ownership(content, examples[i].color));
    out.winner = winner as c_int;
    out.number = i as c_int;
    out.komi = examples[i].board.komi();

    match examples[i].policy {
        Some(ref policy) => {
            assert_eq!(policy.len(), 905, "illegal policy -- {:?}", policy);
            out.policy.copy_from_slice(&b85::decode::<f16, f32>(policy).unwrap());
        },
        None => {
            out.policy.copy_from_slice(&EMPTY_POLICY);
        }
    };

    match next_example.and_then(|example| example.policy) {
        Some(ref policy) => {
            assert_eq!(policy.len(), 905, "illegal next_policy -- {:?}", policy);
            out.next_policy.copy_from_slice(&b85::decode::<f16, f32>(policy).unwrap());
        },
        None => {
            out.next_policy.copy_from_slice(&EMPTY_POLICY);
        }
    };

    0
}

/// Returns the komi of the given SGF, as parsed by a simple regular expression.
/// 
/// # Arguments
/// 
/// * - `content` - 
/// 
fn get_komi_from_sgf(content: &str) -> Result<f32, i32> {
    lazy_static! {
        static ref KOMI: Regex = Regex::new(r"KM\[([^\]]*)\]").unwrap();
    }

    if let Some(caps) = KOMI.captures(&content) {
        if caps[1] == *"0" || caps[1] == *"0.0" {
            Ok(DEFAULT_KOMI)  // Fox sometimes output an empty komi
        } else {
            match caps[1].parse::<f32>() {
                Ok(komi) => {
                    if komi >= 100.0 {
                        // Fox seems to sometimes output 550 instead of 5.5, etc.
                        Ok(komi / 100.0)
                    } else {
                        Ok(komi)
                    }
                },
                Err(_) => { return Err(-21); },
            }
        }
    } else {
        Ok(DEFAULT_KOMI)
    }
}

/// Returns the winner of the given SGF, as parsed by a simple regular expression.
/// 
/// # Arguments
/// 
/// * - `content` - 
/// 
fn get_winner_from_sgf(content: &str) -> Result<Color, i32> {
    lazy_static! {
        static ref WINNER: Regex = Regex::new(r"RE\[([^\]]+)\]").unwrap();
    }

    if let Some(caps) = WINNER.captures(&content) {
        match caps[1].chars().nth(0) {
            Some('B') => Ok(Color::Black),
            Some('W') => Ok(Color::White),
            _ => Err(-22)
        }
    } else {
        Err(-22)
    }
}

/// Returns if the given SGF has given score.
/// 
/// # Arguments
/// 
/// * `content` - 
/// 
fn is_scored(content: &str) -> bool {
    lazy_static! {
        static ref SCORED: Regex = Regex::new(r"RE\[[BW]\+[0-9\.]+\]").unwrap();
    }

    SCORED.is_match(content) 
}

/// Update the vertex ownership based on the given SGF properties.
/// 
/// # Arguments
/// 
/// * `property` - 
/// * `value` - 
/// * `ownership` - 
/// 
fn set_vertex_ownerships(property: Option<Captures>, value: f32, ownership: &mut [f32]) {
    lazy_static! {
        static ref VERTICES: Regex = Regex::new(r"\[([a-z]*)\]").unwrap();
    }

    if let Some(tb) = property {
        for vertex in VERTICES.captures_iter(tb.get(1).unwrap().as_str()) {
            if let Ok(point) = CGoban::parse(vertex.get(1).unwrap().as_str()) {
                if point != Point::default() {
                    ownership[point.to_packed_index()] = value;
                }
            }
        }
    }
}

/// Returns a list that indicates who owns each vertex of the board at the end of the game.
///
/// # Arguments
///
/// * `content` -
/// * `to_move` -
///
fn get_vertex_ownership(content: &str, to_move: Color) -> Vec<f32> {
    lazy_static! {
        static ref TB: Regex = Regex::new(r"TB((?:[\s\r\n]*\[(?:[a-z]*)\])+)").unwrap();
        static ref TW: Regex = Regex::new(r"TW((?:[\s\r\n]*\[(?:[a-z]*)\])+)").unwrap();
    }

    let mut ownership = vec! [0.0; 361];
    set_vertex_ownerships(TB.captures(content), if to_move == Color::Black { 1.0 } else { -1.0 }, &mut ownership);
    set_vertex_ownerships(TW.captures(content), if to_move == Color::Black { -1.0 } else { 1.0 }, &mut ownership);

    ownership
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resign_is_not_scored() {
        assert!(!is_scored(&"(;GM[1]RE[B+R])"));
    }

    #[test]
    fn scored_is_scored() {
        assert!(is_scored(&"(;GM[1]RE[B+1.5])"));
        assert!(is_scored(&"(;GM[1]RE[W+1.5])"));
    }

    #[test]
    fn black_is_winner() {
        assert_eq!(get_winner_from_sgf(&"(;GM[1]RE[B+0.5])"), Ok(Color::Black));
    }

    #[test]
    fn white_is_winner() {
        assert_eq!(get_winner_from_sgf(&"(;GM[1]RE[W+0.5])"), Ok(Color::White));
    }

    #[test]
    fn komi_is() {
        assert_eq!(get_komi_from_sgf(&"(;GM[1]KM[0])"), Ok(DEFAULT_KOMI));
        assert_eq!(get_komi_from_sgf(&"(;GM[1]KM[7.5])"), Ok(7.5));
        assert_eq!(get_komi_from_sgf(&"(;GM[1]KM[6.5])"), Ok(6.5));
        assert_eq!(get_komi_from_sgf(&"(;GM[1]KM[5.5])"), Ok(5.5));
        assert_eq!(get_komi_from_sgf(&"(;GM[1]KM[0.5])"), Ok(0.5));
    }

    #[test]
    fn territory() {
        let content = "(GM[1];TB[aa][ba]TW[da][ea])";
        let ownership = get_vertex_ownership(content, Color::Black);

        for i in 0..361 {
            if i == 0 || i == 1 {
                assert_eq!(ownership[i], 1.0, "ownership[{}] != 1.0", i);
            } else if i == 3 || i == 4 {
                assert_eq!(ownership[i], -1.0, "ownership[{}] != -1.0", i);
            } else {
                assert_eq!(ownership[i], 0.0, "ownership[{}] != 0.0", i);
            }
        }
    }
}
