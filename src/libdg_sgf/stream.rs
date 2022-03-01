// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use super::lex_token::*;
use super::lex::*;
use super::sgf_token::*;

pub struct Stream<'a> {
    input: &'a [u8],
    lex: Lex<'a>,
    buf: Vec<LexToken>,
    last_property: &'a [u8],
    num_rparen: usize
}

impl<'a> Stream<'a> {
    pub fn new(s: &'a [u8]) -> Self {
        Self {
            input: s,
            lex: Lex::lex(s),
            buf: Vec::with_capacity(8),
            last_property: &s[0..0],
            num_rparen: 0
        }
    }

    fn try_fill_buf(&mut self) -> bool {
        if let Some(tok) = self.lex.next() {
            self.buf.push(tok);
            true
        } else {
            false
        }
    }

    fn is_main_path(&self) -> bool {
        self.num_rparen == 0
    }

    fn skip_until_rparen(&mut self) {
        loop {
            match self.lex.next() {
                Some(LexToken::LParen) => { self.skip_until_rparen() },
                Some(LexToken::RParen) => { break },
                Some(_) => { /* pass */ },
                None => break
            }
        }
    }

    fn parse_token(&self, property: &'a [u8], value: &'a [u8]) -> Option<SgfToken<'a>> {
        if property == b"B" || property == b"W" {
            Some(SgfToken::Play { color: property, point: value })
        } else if property == b"AB" || property == b"AW" {
            Some(SgfToken::Add { color: &property[1..], point: value })
        } else if property == b"TB" || property == b"TW" {
            Some(SgfToken::Territory { color: &property[1..], point: value })
        } else if property == b"RE" {
            Some(SgfToken::Result { text: value })
        } else if property == b"KM" {
            Some(SgfToken::Komi { text: value })
        } else if property == b"HA" {
            Some(SgfToken::Handicap { text: value })
        } else if property == b"SZ" {
            Some(SgfToken::Size { text: value })
        } else {
            None
        }
    }
}

impl<'a> Iterator for Stream<'a> {
    type Item = SgfToken<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.try_fill_buf() {
            let (token, consume) = match self.buf[..] {
                [LexToken::LParen] if self.is_main_path() => { (None, true) },
                [LexToken::LParen] => { self.skip_until_rparen(); (None, true) },
                [LexToken::RParen] => { self.num_rparen += 1; (None, true) },
                [LexToken::SemiColon] => { (Some(SgfToken::Node), true) },
                [LexToken::Text { offset, len }] => {
                    self.last_property = &self.input[offset..(offset+len)];
                    (None, true)
                },
                [LexToken::LBrack, LexToken::RBrack] => {
                    (self.parse_token(self.last_property, &self.input[0..0]), true)
                },
                [LexToken::LBrack, LexToken::Text { offset, len }, LexToken::RBrack] => {
                    let value = &self.input[offset..(offset+len)];
                    (self.parse_token(self.last_property, value), true)
                },
                _ => { (None, false) }
            };

            if consume {
                self.buf.clear();
            }

            if token.is_some() {
                return token;
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_main_path() {
        let s = b"(;GM[1]FF[4]KM[7.5]SZ[19];B[pd](;W[dd];B[dp];W[pp];B[cf];W[fc](;B[bd];W[cc])(;B[cd])(;B[cc])(;B[hc];W[ch]))(;W[pp];B[dp]))";
        let stream = Stream::new(s);

        assert_eq!(
            stream.collect::<Vec<_>>(),
            vec! [
                SgfToken::Node,
                SgfToken::Komi { text: b"7.5" },
                SgfToken::Size { text: b"19" },
                SgfToken::Node,
                SgfToken::Play { color: b"B", point: b"pd" },
                SgfToken::Node,
                SgfToken::Play { color: b"W", point: b"dd" },
                SgfToken::Node,
                SgfToken::Play { color: b"B", point: b"dp" },
                SgfToken::Node,
                SgfToken::Play { color: b"W", point: b"pp" },
                SgfToken::Node,
                SgfToken::Play { color: b"B", point: b"cf" },
                SgfToken::Node,
                SgfToken::Play { color: b"W", point: b"fc" },
                SgfToken::Node,
                SgfToken::Play { color: b"B", point: b"bd" },
                SgfToken::Node,
                SgfToken::Play { color: b"W", point: b"cc" },
            ]
        );
    }

    #[test]
    fn multiple_values() {
        let s = b"(;AB[pd][dd][dp][pp];W[aa])";
        let stream = Stream::new(s);

        assert_eq!(
            stream.collect::<Vec<_>>(),
            vec! [
                SgfToken::Node,
                SgfToken::Add { color: b"B", point: b"pd" },
                SgfToken::Add { color: b"B", point: b"dd" },
                SgfToken::Add { color: b"B", point: b"dp" },
                SgfToken::Add { color: b"B", point: b"pp" },
                SgfToken::Node,
                SgfToken::Play { color: b"W", point: b"aa" },
            ]
        );
    }

    #[test]
    fn unicode() {
        let s = "(;GM[1]FF[4]SZ[19]GN[]DT[2015-06-18]PB[å°åä¹é¸]PW[é£éªå£«]BR[9æ®µ]WR[9æ®µ]KM[0]HA[0]RU[Japanese]AP[GNU Go:3.8]RE[B+R]TM[60]TC[1]TT[15];B[dp];W[dd];B[qp];W[pd];B[qf];W[nc];B[pi];W[nq];B[op];W[pr];B[mp];W[qq];B[rp];W[np];B[no];W[mq];B[lp];W[lq];B[kp];W[oo];B[nn];W[on];B[om];W[pn];B[pq];W[qr];B[pm];W[qn];B[rm];W[qm];B[ql];W[rn];B[rl];W[rq];B[sp];W[sn];B[kq];W[nm];B[ok];W[cn];B[fp];W[dj];B[fc];W[cf];B[jd];W[re];B[el];W[en];B[cl];W[gn];B[gl];W[hp];B[ho];W[go];B[gp];W[io];B[hq];W[hn];B[iq];W[bp];B[cq];W[il];B[fj];W[cj];B[hj];W[jj];B[jk];W[ik];B[ij];W[kk];B[ji];W[jl];B[ki];W[rf];B[fe];W[kc];B[jc];W[kd];B[ke];W[jb];B[ib];W[kb];B[cc];W[dc];B[db];W[cd];B[bb];W[bc];B[cb];W[je];B[kf];W[gd];B[fd];W[ic];B[id];W[hc];B[ie];W[hb];B[jn];W[ln];B[mm];W[nl];B[ml];W[nk];B[mk];W[nj];B[mj];W[ni];B[do];W[dn];B[bm];W[jo];B[mi];W[pj];B[km];W[lm];B[ll];W[kl];B[kn];W[ko];B[lo];W[in];B[mn];W[bn];B[oj];W[pl];B[ol];W[pk];B[oi];W[qj];B[qi];W[rj];B[ri];W[sm];B[bk];W[ej];B[ek];W[bq];B[ch];W[dh];B[bi];W[bj];B[aj];W[cg];B[bg];W[fi];B[gi];W[ef];B[fh];W[ci];B[bh];W[eh];B[ei];W[bf];B[eg];W[dg];B[ee];W[de];B[fg];W[ae];B[ac];W[ai];B[ah];W[ag];B[af];W[gf];B[ff];W[ag];B[br];W[al];B[af];W[ak];B[am];W[ai];B[bl];W[ip];B[aj];W[ag];B[hl];W[im];B[af];W[nh];B[ng];W[ag];B[sl];W[sk];B[af];W[oh];B[pg];W[ag];B[si];W[rk];B[af];W[og];B[of];W[ag];B[or];W[oq];B[af];W[nf];B[mg];W[ag];B[pp];W[nr];B[af];W[eb];B[be];W[ec];B[fb])";
        let s = s.as_bytes();
        let stream = Stream::new(s);

        assert_eq!(stream.count(), 423)
    }
}
