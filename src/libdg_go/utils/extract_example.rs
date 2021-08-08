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

use super::features::{self, HWC, LzFeatures, Features};
use super::sgf::{Sgf, SgfEntry, SgfError, get_komi_from_sgf, is_scored, get_winner_from_sgf};
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
    pub color: c_int,
    pub policy: [f32; 362],
    pub ownership: [f32; 361],
    pub winner: c_int,
    pub number: c_int,
    pub komi: f32,
    pub lz_features: [f16; 6498],
    pub features: [f16; features::Default::size()]
}

#[repr(i32)]
#[derive(Copy, Clone)]
pub enum Kind {
    Default = 0,
    Lz = 1
}

impl Kind {
    fn extract(&self, examples: &[Candidate], i: usize) -> Vec<f16> {
        match *self {
            Kind::Default => {
                features::Default::new(&examples[i].board).get_features::<HWC, f16>(
                    examples[i].color,
                    symmetry::Transform::Identity
                )
            },

            Kind::Lz => {
                let mut board_history: Vec<&Board> =
                    if i < 8 {
                        examples.iter().take(i + 1).map(|cand| &cand.board).collect()
                    } else {
                        let start = (i as i32 - 7).max(0) as usize;
                        examples.iter().skip(start).take(8).map(|cand| &cand.board).collect()
                    };
                board_history.reverse();

                LzFeatures::new(board_history).get_features::<HWC, f16>(
                    examples[i].color,
                    symmetry::Transform::Identity
                )
            }
        }
    }
}

impl Default for Example {
    fn default() -> Example {
        Example {
            index: 0,
            color: 0,
            policy: [f32::from(0.0); 362],
            ownership: [f32::from(0.0); 361],
            winner: 0,
            number: 0,
            komi: DEFAULT_KOMI,
            lz_features: [f16::from(0.0); 6498],
            features: [f16::from(0.0); features::Default::size()]
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
            value: m.value.map(|v| (v + 1.0) / 2.0)
        }
    }
}

lazy_static! {
    static ref RNG: Mutex<StdRng> = Mutex::new(StdRng::from_entropy());
}

/// Returns the number of features used internally.
#[no_mangle]
pub unsafe extern fn get_num_features() -> c_int {
    features::Default::num_features() as i32
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
/// - `kind` - The kind of features to extract.
/// - `out` - Output of the extracted example.
///
#[no_mangle]
pub unsafe extern fn extract_single_example(
    raw_sgf_content: *const c_char,
    out: *mut Example,
    num_examples: c_int
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

        choose_examples(&examples, has_policy, num_examples as usize).map(|i| {
            copy_candidates_to(content, &examples, i, num_examples as usize, out)
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
fn choose_examples(examples: &[Candidate], has_policy: bool, num_examples: usize) -> Option<usize> {
    let max_examples = if examples.len() > num_examples { examples.len() - num_examples } else { 0 };
    let candidate_examples: Vec<usize> = (0..max_examples)
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
                Some(val) => 0.6 - (val - 0.5).abs(),
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

/// Copy the `i`:th features, and correct future values, in addition to the next
/// `num_examples` from the given SGF, into the given `Example`'s array.
///
/// # Arguments
///
/// * `content` -
/// * `examples` -
/// * `i` -
/// * `num_examples` -
/// * `out` -
///
unsafe fn copy_candidates_to(
    content: &str,
    examples: &[Candidate],
    i: usize,
    num_examples: usize,
    out: *mut Example
) -> c_int
{
    for j in 0..num_examples {
        let result = copy_candidate_to(content, &examples, i + j, &mut *out.add(j));

        if result != 0 {
            return result;
        }
    }

    0
}

/// Copy the `i`:th feature, and correct future value from the given SGF, into
/// the given `Example`.
///
/// # Arguments
///
/// * `content` -
/// * `kind` -
/// * `examples` -
/// * `i` -
/// * `out` -
///
fn copy_candidate_to(
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
    let lz_features = Kind::Lz.extract(examples, i);
    let features = Kind::Default.extract(examples, i);

    out.lz_features.clone_from_slice(&lz_features);
    out.features.clone_from_slice(&features);
    out.index = examples[i].index as c_int;
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

    0
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
