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

use ::{DEFAULT_KOMI, Board, Color, Point};
use memchr::memchr;
use regex::Regex;

static SGF_LETTERS: [char; 26] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
];

pub trait SgfCoordinate {
    fn to_sgf(p: Point) -> String;
    fn parse(s: &str) -> Result<Point, SgfCoordinateError>;
}

pub enum SgfCoordinateError {
    InvalidLen,
    UnrecognizedCharacter
}

pub struct CGoban;

impl SgfCoordinate for CGoban {
    fn to_sgf(point: Point) -> String {
        if point != Point::default() {
            format!("{}{}", SGF_LETTERS[point.x()], SGF_LETTERS[point.y()])
        } else {
            String::new()
        }
    }

    fn parse(s: &str) -> Result<Point, SgfCoordinateError> {
        if s.is_empty() {
            Ok(Point::default())
        } else if s.len() == 2 {
            let mut ch = s.chars();
            let x = ch.next().and_then(|x| { SGF_LETTERS.binary_search(&x).ok() });
            let y = ch.next().and_then(|y| { SGF_LETTERS.binary_search(&y).ok() });

            match (x, y) {
                (Some(x), Some(y)) => {
                    if x >= 19 || y >= 19 {
                        Ok(Point::default())
                    } else {
                        Ok(Point::new(x, y))
                    }
                },
                _ => Err(SgfCoordinateError::UnrecognizedCharacter)
            }
        } else {
            Err(SgfCoordinateError::InvalidLen)
        }
    }
}

pub struct Sabaki;

impl SgfCoordinate for Sabaki {
    fn to_sgf(point: Point) -> String {
        if point != Point::default() {
            format!("{}{}", SGF_LETTERS[point.x()], SGF_LETTERS[18 - point.y()])
        } else {
            String::new()
        }
    }

    fn parse(s: &str) -> Result<Point, SgfCoordinateError> {
        match CGoban::parse(s) {
            Ok(point) if point != Point::default() => {
                Ok(Point::new(point.x(), 18 - point.y()))
            },
            err => err
        }
    }
}

// -------- Parsing --------

#[derive(Debug)]
pub enum SgfError {
    IllegalMove,
    ParseError
}

pub struct SgfEntry<'a> {
    pub board: Board,
    pub policy: Option<&'a [u8]>,
    pub value: Option<f32>,

    pub color: Color,
    pub point: Point
}

pub struct Sgf<'a> {
    content: &'a [u8],
    board: Vec<Board>,
    index: usize,
}

struct SgfMatch<'a> {
    color: Color,
    point: Point,

    policy: Option<&'a [u8]>,
    value: Option<f32>,

    begin: usize
}

fn skip_ws(bytes: &[u8], start_at: &mut usize) {
    while *start_at < bytes.len() {
        let ch = bytes[*start_at];

        if ch != 32 && ch != 9 && ch != 10 && ch != 13 {
            break
        }

        *start_at += 1;
    }
}

fn skip_until_next<'a>(bytes: &'a [u8], start_at: &mut usize, goal: u8) -> &'a [u8] {
    let starting_point = *start_at;

    memchr(goal, &bytes[starting_point..])
        .map(|i| {
            *start_at += i + 1;

            &bytes[starting_point..(starting_point + i)]
        })
        .unwrap_or_else(|| &bytes[0..0])
}

fn peek_forward2(bytes: &[u8], at_index: usize, peek_1: u8, peek_2: u8) -> bool {
    at_index < bytes.len() - 2 && bytes[at_index] == peek_1 && bytes[at_index+1] == peek_2
}

fn find_next_property<'a, 'b>(bytes: &'a [u8], start_at: &mut usize) -> Option<(&'b [u8], &'b [u8])>
    where 'a: 'b
{
    skip_ws(bytes, start_at);

    let key = skip_until_next(bytes, start_at, b'[');
    let value = skip_until_next(bytes, start_at, b']');

    if key.is_empty() {
        None
    } else {
        Some((key, value))
    }
}

fn find_next_vertex(bytes: &[u8], start_at: &mut usize) -> Option<(Color, Point)> {
    match find_next_property(bytes, start_at) {
        None => None,
        Some((key, value)) => {
            if let Some(color) = ::std::str::from_utf8(key).ok().and_then(|x| x.trim().parse::<Color>().ok()) {
                let point = ::std::str::from_utf8(value).ok().and_then(|x| CGoban::parse(x).ok()).unwrap_or(Point::default());

                Some((color, point))
            } else {
                None
            }
        }
    }
}

/// Returns the next occurence of a vertex in the given SGF file, where a
/// vertex is denoted by the pattern `;[BW]\[(...)\]`.
///
/// # Arguments
///
/// * `bytes` -
/// * `start_at` -
///
fn find_next_move<'a>(bytes: &'a [u8], start_at: &mut usize) -> Option<SgfMatch<'a>> {
    while *start_at < bytes.len() - 4 {
        if bytes[*start_at] == b';' {
            let starting_index = *start_at;

            *start_at += 1;
            if let Some((color, point)) = find_next_vertex(bytes, start_at) {
                skip_ws(bytes, start_at);
                let policy = if peek_forward2(bytes, *start_at, b'P', b'[') {
                    find_next_property(bytes, start_at).and_then(|x| {
                        Some(x.1)
                    })
                } else {
                    None
                };

                skip_ws(bytes, start_at);
                let value = if peek_forward2(bytes, *start_at, b'V', b'[') {
                    find_next_property(bytes, start_at).and_then(|x| {
                        ::std::str::from_utf8(x.1).ok().and_then(|x| x.parse::<f32>().ok())
                    })
                } else {
                    None
                };

                return Some(SgfMatch {
                    color: color,
                    point: point,

                    policy: policy,
                    value: value,

                    begin: starting_index
                });
            }
        } else {
            *start_at += 1;
        }
    }

    None
}

impl<'a> Sgf<'a> {
    pub fn new(content: &'a [u8], komi: f32) -> Sgf {
        Sgf {
            content: content,
            board: vec! [Board::new(komi)],
            index: 0
        }
    }
}

impl<'a> Iterator for Sgf<'a> {
    type Item = Result<SgfEntry<'a>, SgfError>;

    fn next(&mut self) -> Option<Result<SgfEntry<'a>, SgfError>> {
        let starting_index = if self.index == 0 { 0 } else { self.index - 1 };

        if let Some(m) = find_next_move(self.content, &mut self.index) {
            // unwind the stack for the nested game tree
            let mut in_property = false;

            for i in starting_index..m.begin {
                if i == 0 || self.content[i-1] != b'\'' {
                    if self.content[i] == b'[' {
                        in_property = true;
                    } else if self.content[i] == b']' {
                        in_property = false;
                    } else if !in_property && self.content[i] == b'(' {
                        if self.board.is_empty() {
                            return Some(Err(SgfError::ParseError));
                        }

                        let prev_board = self.board.last().unwrap().clone();

                        self.board.push(prev_board);
                    } else if !in_property && self.content[i] == b')' {
                        if self.board.is_empty() {
                            return Some(Err(SgfError::ParseError));
                        }

                        self.board.pop();
                    }
                }
            }

            // if we have a valid, or pass, move then advance the board state
            let board = self.board.last_mut().unwrap();
            let prev_board = board.clone();

            if m.point != Point::default() {
                if board.is_valid(m.color, m.point) {
                    board.place(m.color, m.point);
                } else {
                    return Some(Err(SgfError::IllegalMove));
                }
            }

            Some(Ok(SgfEntry {
                board: prev_board,
                policy: m.policy,
                value: m.value,

                color: m.color,
                point: m.point,
            }))
        } else {
            None
        }
    }
}

/// Returns the komi of the given SGF, as parsed by a simple regular expression.
///
/// # Arguments
///
/// * - `content` -
///
pub fn get_komi_from_sgf(content: &str) -> Result<f32, i32> {
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
pub fn get_winner_from_sgf(content: &str) -> Result<Color, i32> {
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
pub fn is_scored(content: &str) -> bool {
    lazy_static! {
        static ref SCORED: Regex = Regex::new(r"RE\[[BW]\+[0-9\.]+\]").unwrap();
    }

    SCORED.is_match(content)
}

#[cfg(test)]
mod tests {
    use test::{black_box, Bencher};
    use super::*;

    #[test]
    fn simple_sgf() {
        let moves = Sgf::new(b"(;B[dp];W[dd])", 0.5)
            .map(|x| x.ok().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(moves.len(), 2);
        assert_eq!(moves[0].board.count(), 0);
        assert_eq!(moves[0].board.komi(), 0.5);
        assert_eq!(moves[0].color, Color::Black);
        assert_eq!(moves[0].point, Point::new(3, 15));
        assert_eq!(moves[1].board.count(), 1);
        assert_eq!(moves[1].color, Color::White);
        assert_eq!(moves[1].point, Point::new(3, 3));
    }

    #[test]
    fn rparen_sgf() {
        let moves = Sgf::new(b"(;B[dp]C[)))];W[dd])", 0.5)
            .map(|x| x.ok().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(moves.len(), 2);
    }

    #[test]
    fn branching_sgf() {
        let moves = Sgf::new(b" ( ;B[dp] ( ; W [pd] ) ( ; W [dd] ) ( ; W[pp] ) ) ", 7.5)
            .map(|x| x.ok().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(moves.len(), 4);
        assert_eq!(moves[0].board.count(), 0);
        assert_eq!(moves[0].board.komi(), 7.5);
        assert_eq!(moves[0].color, Color::Black);
        assert_eq!(moves[0].point, Point::new(3, 15));
        assert_eq!(moves[1].board.count(), 1);
        assert_eq!(moves[1].color, Color::White);
        assert_eq!(moves[1].point, Point::new(15, 3));
        assert_eq!(moves[2].board.count(), 1);
        assert_eq!(moves[2].color, Color::White);
        assert_eq!(moves[2].point, Point::new(3, 3));
        assert_eq!(moves[3].board.count(), 1);
        assert_eq!(moves[3].color, Color::White);
        assert_eq!(moves[3].point, Point::new(15, 15));
    }

    #[test]
    fn nested_sgf() {
        let moves = Sgf::new(b"(;B[dp](;W[dd];B[pd])(;W[qp];B[dd](;W[pd](;B[oq])(;B[op];W[oq]))(;W[np])))", 7.5)
            .map(|x| x.unwrap())
            .collect::<Vec<_>>();

        assert_eq!(moves.len(), 10);
        assert_eq!(moves[0].board.count(), 0);
        assert_eq!(moves[0].color, Color::Black);
        assert_eq!(moves[1].board.count(), 1);
        assert_eq!(moves[1].color, Color::White);
        assert_eq!(moves[2].board.count(), 2);
        assert_eq!(moves[2].color, Color::Black);
        assert_eq!(moves[3].board.count(), 1);
        assert_eq!(moves[3].color, Color::White);
        assert_eq!(moves[4].board.count(), 2);
        assert_eq!(moves[4].color, Color::Black);
        assert_eq!(moves[5].board.count(), 3);
        assert_eq!(moves[5].color, Color::White);
        assert_eq!(moves[6].board.count(), 4);
        assert_eq!(moves[6].color, Color::Black);
        assert_eq!(moves[7].board.count(), 4);
        assert_eq!(moves[7].color, Color::Black);
        assert_eq!(moves[8].board.count(), 5);
        assert_eq!(moves[8].color, Color::White);
        assert_eq!(moves[9].board.count(), 3);
        assert_eq!(moves[9].color, Color::White);
    }

    #[bench]
    fn bench_sgf(b: &mut Bencher) {
        let sgf = black_box(r#"
(;FF[4]GM[1]SZ[19]CA[UTF-8]SO[gokifu.com]BC[cn]WC[cn]EV[]
PB[芈昱廷]BR[9p]PW[江维杰]WR[9p]KM[7.5]
DT[2018-12-03]RE[W+R]TM[60]LT[]LC[1]GK[1]
;B[pd];W[dp];B[pp];W[dd];B[cq];W[cp];B[dq];W[fq];B[fr];W[gq]
;B[eq];W[ep];B[gr];W[hq];B[cf];W[fc];B[dj];W[qc];B[qd];W[pc]
;B[od];W[nb];B[qj];W[mc];B[md];W[ld];B[le];W[nd];B[me];W[ne]
;B[oc];W[ob];B[nc];W[nf];B[lc];W[mb];B[kd];W[qf];B[rc];W[rb]
;B[pf];W[pg];B[qg];W[rf];B[of];W[og];B[ng];W[mf];B[ph];W[oh]
;B[rd];W[nh];B[sb];W[qh];B[qb];W[kb];B[kg];W[lg];B[gc];W[pb]
;B[ra];W[ib];B[fb];W[fd];B[gd];W[id];B[eb];W[df];B[ie];W[hb]
;B[jc];W[gb];B[fe];W[dc];B[jb];W[he];B[ge];W[hd];B[lb];W[je]
;B[lh];W[ce];B[db];W[cb];B[dg];W[ef];B[gg];W[ig];B[ee];W[de]
;B[eg];W[cg];B[bf];W[fg];B[ch];W[fh];B[kh];W[qn];B[nj];W[mh]
;B[fo];W[cl];B[fp];W[bj];B[dn];W[bq];B[br];W[cn];B[dl];W[dm]
;B[em];W[cm];B[el];W[ho];B[bp];W[bo];B[aq];W[en];B[eo];W[do]
;B[fn];W[dn];B[lq];W[hm];B[gm];W[hl];B[bi];W[ej];B[dk];W[bk]
;B[di];W[be];B[ip];W[hp];B[bg];W[ql];B[gh];W[fi];B[gi];W[gj]
;B[hj];W[gk];B[ff];W[ij];B[ii];W[hk];B[hi];W[jj];B[ji];W[lj]
;B[li];W[lf];B[ke];W[lk];B[kf];W[ja];B[jl];W[km];B[jm];W[kn]
;B[in];W[jn];B[im];W[hn];B[io];W[jk];B[mi];W[oj];B[ok];W[pj]
;B[pk];W[qk];B[pm];W[om];B[nm];W[on];B[nn];W[kp];B[iq];W[nl]
;B[ml];W[no];B[mo];W[mp];B[oo];W[np];B[pn];W[hr];B[ir];W[is]
;B[js];W[hs];B[nr];W[mr];B[mq];W[nq];B[lr];W[or];B[ms];W[jr]
;B[kq];W[ks];B[lp];W[mn];B[mm];W[lo];B[oq];W[jq];B[ko];W[jo]
;B[jp];W[ko];B[qm];W[rm];B[rn];W[lm];B[rj];W[qi];B[op];W[mo]
;B[pr];W[la];B[cc];W[bc];B[bb];W[cd];B[ca];W[cc];B[sk];W[ri]
;B[rl];W[mk];B[ol];W[nk];B[ih];W[pa];B[ad];W[ab];B[ba];W[ac]
;B[ae];W[fa])"#.as_bytes());

        b.iter(move || {
            let mut last_num_played = -1;
            let mut last_color = Color::White;
            let mut count = 0;

            for entry in Sgf::new(sgf, 0.5) {
                let entry = entry.ok().unwrap();

                debug_assert!(entry.board.count() as i32 > last_num_played);
                debug_assert_ne!(entry.color, last_color);

                last_color = entry.color;
                last_num_played = entry.board.count() as i32;
                count += 1;

                // consume the entire entry to avoid compiler optimizations
                black_box(entry);
            }

            assert_eq!(count, 242);
        })
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
}
