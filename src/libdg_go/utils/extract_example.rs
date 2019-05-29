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

use dg_utils::types::f16;
use dg_utils::b85;

use libc::{c_char, c_int};
use rand::distributions::{Normal, Distribution};
use rand::rngs::StdRng;
use rand::{FromEntropy, SeedableRng};
use regex::{Regex, Captures};
use std::ffi::CStr;
use std::sync::Mutex;
use utils::sgf::{CGoban, SgfCoordinate};

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

    // this must be at the end since Python does not know its size, and we therefore allocate it
    // dynamically
    pub features: [f32; FEATURE_SIZE],
}

impl Default for Example {
    fn default() -> Example {
        Example {
            features: [f32::from(0.0); FEATURE_SIZE],
            index: 0,
            next_index: 0,
            color: 0,
            policy: [f32::from(0.0); 362],
            next_policy: [f32::from(0.0); 362],
            ownership: [f32::from(0.0); 361],
            winner: 0,
            number: 0,
            komi: DEFAULT_KOMI
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

lazy_static! {
    static ref RNG: Mutex<StdRng> = Mutex::new(StdRng::from_entropy());
}

/// Returns the number of features used internally.
#[no_mangle]
pub unsafe extern "C" fn get_num_features() -> c_int {
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
pub unsafe extern "C" fn set_seed(seed: i32) {
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
pub unsafe extern "C" fn extract_single_example(
    raw_sgf_content: *const c_char,
    out: *mut Example
) -> c_int
{
    lazy_static! {
        static ref EMPTY_POLICY: Vec<f32> = vec! [0.0; 362];

        static ref WINNER: Regex = Regex::new(r"RE\[([^\]]+)\]").unwrap();
        static ref SCORED: Regex = Regex::new(r"RE\[[BW]\+[0-9\.]+\]").unwrap();
        static ref KOMI: Regex = Regex::new(r"KM\[([^\]]*)\]").unwrap();
    }

    CStr::from_ptr(raw_sgf_content as *const _).to_str().map(|content| {
        // find the komi by looking for the pattern `KM[...]` at any point
        // in the file.
        let komi = {
            if let Some(caps) = KOMI.captures(&content) {
                if caps[1] == *"0" || caps[1] == *"0.0" {
                    DEFAULT_KOMI  // Fox sometimes output an empty komi
                } else {
                    match caps[1].parse::<f32>() {
                        Ok(komi) => {
                            if komi >= 100.0 {
                                // Fox seems to sometimes output 550 instead of 5.5, etc.
                                komi / 100.0
                            } else {
                                komi
                            }
                        },
                        Err(_) => { return -21; },
                    }
                }
            } else {
                DEFAULT_KOMI
            }
        };

        // find the winner by looking for the pattern `RE[...]`.
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
        let mut examples = Vec::with_capacity(254);
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

        let chosen_candidate = RNG.lock().map(|ref mut rng| {
            let normal_distr = Normal::new(
                0.5 * candidate_examples.len() as f64,
                0.25 * candidate_examples.len() as f64
            );
            let mut index;

            loop {
                index = normal_distr.sample(&mut **rng).round() as isize;

                if index >= 0 && (index as usize) < candidate_examples.len() {
                    break
                }
            }

            candidate_examples.get(index as usize)
        }).unwrap_or(None);

        chosen_candidate.map(|&i| {
            let next_example = examples.get(i+1);
            let features = examples[i].board.get_features::<HWC, f32>(
                examples[i].color,
                symmetry::Transform::Identity
            );

            (*out).features.clone_from_slice(&features);
            (*out).index = examples[i].index as c_int;
            (*out).next_index = next_example.map(|example| example.index).unwrap_or(361) as c_int;
            (*out).color = examples[i].color as c_int;
            (*out).policy.clone_from_slice(&match examples[i].policy {
                Some(ref policy) => {
                    assert_eq!(policy.len(), 905, "illegal policy -- {:?}", policy);

                    b85::decode::<f16, f32>(policy).unwrap()
                },
                None => EMPTY_POLICY.clone()
            });
            (*out).next_policy.clone_from_slice(&match next_example.and_then(|example| example.policy) {
                Some(ref policy) => {
                    assert_eq!(policy.len(), 905, "illegal policy -- {:?}", policy);

                    b85::decode::<f16, f32>(policy).unwrap()
                },
                None => EMPTY_POLICY.clone()
            });
            (*out).ownership.clone_from_slice(&get_vertex_ownership(content, examples[i].color));
            (*out).winner = winner as c_int;
            (*out).number = i as c_int;
            (*out).komi = examples[i].board.komi();

            0
        }).unwrap_or(-30)
    }).unwrap_or(-1) as c_int
}

fn set_vertex_ownerships(property: Option<Captures>, value: f32, ownership: &mut [f32]) {
    lazy_static! {
        static ref VERTICES: Regex = Regex::new(r"\[([a-z]*)\]").unwrap();
    }

    if let Some(tb) = property {
        for vertex in VERTICES.captures_iter(tb.get(1).unwrap().as_str()) {
            if let Ok((x, y)) = CGoban::parse(vertex.get(1).unwrap().as_str()) {
                if x < 19 && y < 19 {
                    let index = 19 * y + x;

                    ownership[index] = value;
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
        let content = r#"
(;GM[1]FF[4]SZ[19]GN[GNU Go 3.8]DT[2013-01-15]PB[comeone]PW[DFS]
KM[0.0]HA[6]RU[Japanese]AP[GNU Go:3.8]RE[0]
;W[cn];B[cl];W[em];B[fp];W[dk];B[ck];W[dj];B[co];W[dn];B[bo];W[cf]
;B[bn];W[fc];B[ee];W[hd];B[ch];W[ef];B[ff];W[eg];B[fe];W[hf];B[fg]
;W[eh];B[hg];W[ig];B[hh];W[fh];B[gf];W[if];B[ih];W[ld];B[ke];W[kg]
;B[le];W[nd];B[me];W[pe];B[od];W[ne];B[oe];W[nf];B[lg];W[of];B[qe]
;W[pf];B[qf];W[qg];B[qc];W[jh];B[ij];W[lh];B[mg];W[mh];B[ng];W[oi]
;B[md];W[nc];B[ic];W[lb];B[hc];W[gd];B[gc];W[fd];B[fb];W[eb];B[gb]
;W[dc];B[lc];W[kc];B[mc];W[mb];B[kd];W[cd];B[de];W[ce];B[df];W[cg]
;B[dg];W[dh];B[ci];W[cj];B[bj];W[bk];B[bl];W[bi];B[ak];W[bh];B[il]
;W[qn];B[ob];W[nb];B[kb];W[nq];B[lq];W[oo];B[qq];W[op];B[or];W[gp]
;B[fo];W[jo];B[ip];W[io];B[hp];W[jk];B[ik];W[gh];B[gg];W[hm];B[jm]
;W[ko];B[lm];W[nr];B[qo];W[ro];B[rp];W[po];B[qp];W[nl];B[fl];W[ho]
;B[fm];W[gk];B[hl];W[gl];B[gm];W[gn];B[gj];W[fk];B[nm];W[om];B[ml]
;W[nk];B[mo];W[lp];B[mp];W[mq];B[nn];W[kl];B[kj];W[km];B[kk];W[jl]
;B[kn];W[jn];B[im];W[ln];B[hn];W[go];B[fn];W[jj];B[gq];W[rn];B[qh]
;W[ph];B[rg];W[pg];B[qi];W[pr];B[qr];W[oq];B[sr];W[rs];B[ps];W[os]
;B[pl];W[qk];B[rk];W[pk];B[rl];W[kq];B[jr];W[kr];B[qs];W[sp];B[sq]
;W[rr];B[pn];W[ql];B[so];W[qm];B[nh];W[ni];B[oc];W[ka];B[jb];W[ir]
;B[jq];W[js];B[hr];W[kp];B[iq];W[is];B[hq];W[hs];B[fr];W[fs];B[es]
;W[gs];B[eq];W[el];B[ai];W[ah];B[gi];W[oa];B[pa];W[na];B[qb];W[rm]
;B[rj];W[aj];B[jf];W[jg];B[ai];W[sl];B[sk];W[sm];B[cb];W[ea];B[bc]
;W[ec];B[ja];W[aj];B[ab];W[ba];B[ai];W[fj];B[fi];W[ei];B[qj];W[pj]
;B[ji];W[ki];B[ii];W[aj];B[li]
TB[aa][ba][ca][ga][ha][ia][ja][ka][la][pa][qa][ra][sa][ab][bb][cb]
  [db][fb][gb][jb][kb][ob][pb][qb][rb][sb][ac][bc][gc][hc][ic][jc]
  [kc][lc][mc][oc][qc][rc][sc][ad][bd][dd][ed][id][jd][kd][ld][md]
  [od][qd][rd][sd][de][ee][fe][ge][je][ke][le][me][oe][qe][re][se]
  [df][ff][gf][jf][lf][qf][rf][sf][dg][fg][gg][hg][lg][mg][ng][rg]
  [sg][hh][ih][nh][oh][qh][rh][sh][fi][gi][hi][ii][ji][qi][ri][bj]
  [gj][hj][ij][nj][qj][rj][ak][bk][ck][hk][ik][rk][sk][bl][cl][fl]
  [hl][il][rl][bm][fm][gm][hm][im][jm][bn][en][fn][hn][ao][bo][co]
  [fo][no][ap][bp][cp][dp][ep][fp][hp][ip][aq][bq][cq][dq][eq][fq]
  [gq][hq][iq][jq][ar][br][cr][dr][er][fr][gr][hr][jr][as][bs][cs]
  [ds][es]
TW[da][ea][fa][ma][na][oa][eb][ib][lb][mb][nb][cc][dc][ec][fc][nc]
  [pc][cd][fd][gd][hd][nd][pd][ae][be][ce][he][ie][ne][pe][af][bf]
  [cf][ef][hf][if][kf][mf][nf][of][pf][ag][bg][cg][eg][ig][jg][kg]
  [og][pg][qg][ah][bh][ch][dh][eh][fh][gh][jh][kh][lh][mh][ph][ai]
  [bi][ci][di][ei][ki][li][mi][ni][oi][pi][aj][cj][dj][ej][fj][jj]
  [kj][lj][mj][pj][sj][dk][ek][fk][gk][jk][kk][lk][mk][nk][ok][pk]
  [qk][dl][el][gl][jl][kl][ll][ml][nl][ol][pl][ql][sl][am][cm][dm]
  [em][km][lm][mm][nm][om][pm][qm][rm][sm][an][cn][dn][gn][in][jn]
  [kn][ln][mn][nn][on][pn][qn][rn][sn][do][eo][go][ho][io][jo][ko]
  [lo][mo][oo][po][qo][ro][so][gp][jp][kp][lp][mp][np][op][pp][qp]
  [rp][sp][kq][lq][mq][nq][oq][pq][qq][rq][sq][ir][kr][lr][mr][nr]
  [or][pr][qr][rr][sr][fs][gs][hs][is][js][ks][ls][ms][ns][os][ps]
  [qs][rs][ss]
)"#;

        assert_eq!(get_vertex_ownership(content, Color::Black), vec! [
            1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0,
            -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0,
            -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
            -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
            1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
            1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
            -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
            -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
            1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
            1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
            -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 0.0,
            -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
            -1.0, -1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 1.0,
            -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
            1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            1.0, 1.0, 0.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
            1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
            -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
            1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
        ]);
    }
}
