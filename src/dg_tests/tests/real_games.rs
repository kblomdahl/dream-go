// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

extern crate dg_go;
#[macro_use] extern crate lazy_static;
extern crate regex;

mod common;

use common::playout_game;

/// Famous game for basic sanity checks
#[test]
fn lee_sedol_alpha_go_game_4() {
    let board = playout_game(r#"
;B[pd];W[dp];B[cd];W[qp];B[op];W[oq];B[nq];W[pq];B[cn];W[fq]
;B[mp];W[po];B[iq];W[ec];B[hd];W[cg];B[ed];W[cj];B[dc];W[bp]
;B[nc];W[qi];B[ep];W[eo];B[dk];W[fp];B[ck];W[dj];B[ej];W[ei]
;B[fi];W[eh];B[fh];W[bj];B[fk];W[fg];B[gg];W[ff];B[gf];W[mc]
;B[md];W[lc];B[nb];W[id];B[hc];W[jg];B[pj];W[pi];B[oj];W[oi]
;B[ni];W[nh];B[mh];W[ng];B[mg];W[mi];B[nj];W[mf];B[li];W[ne]
;B[nd];W[mj];B[lf];W[mk];B[me];W[nf];B[lh];W[qj];B[kk];W[ik]
;B[ji];W[gh];B[hj];W[ge];B[he];W[fd];B[fc];W[ki];B[jj];W[lj]
;B[kh];W[jh];B[ml];W[nk];B[ol];W[ok];B[pk];W[pl];B[qk];W[nl]
;B[kj];W[ii];B[rk];W[om];B[pg];W[ql];B[cp];W[co];B[oe];W[rl]
;B[sk];W[rj];B[hg];W[ij];B[km];W[gi];B[fj];W[jl];B[kl];W[gl]
;B[fl];W[gm];B[ch];W[ee];B[eb];W[bg];B[dg];W[eg];B[en];W[fo]
;B[df];W[dh];B[im];W[hk];B[bn];W[if];B[gd];W[fe];B[hf];W[ih]
;B[bh];W[ci];B[ho];W[go];B[or];W[rg];B[dn];W[cq];B[pr];W[qr]
;B[rf];W[qg];B[qf];W[jc];B[gr];W[sf];B[se];W[sg];B[rd];W[bl]
;B[bk];W[ak];B[cl];W[hn];B[in];W[hp];B[fr];W[er];B[es];W[ds]
;B[ah];W[ai];B[kd];W[ie];B[kc];W[kb];B[gk];W[ib];B[qh];W[rh]
;B[qs];W[rs];B[oh];W[sl];B[of];W[sj];B[ni];W[nj];B[oo];W[jp]
"#, None);

    assert_eq!(board.zobrist_hash(), 8306876591293505520, "wrong hash\n{}", board);
}

/// Famous game for basic sanity checks
#[test]
fn ke_jie_alpha_go_game_2() {
    let board = playout_game(r#"
;B[qp];W[pd];B[cq];W[cd];B[ec];W[oq];B[pn];W[df];B[nc];W[qf]
;B[pc];W[qc];B[qb];W[oc];B[pb];W[od];B[ob];W[rc];B[nd];W[mb]
;B[lc];W[lb];B[qd];W[rd];B[jc];W[mc];B[pe];W[oe];B[ld];W[kp]
;B[iq];W[fp];B[dn];W[io];B[ch];W[cl];B[eh];W[cg];B[bg];W[bf]
;B[fn];W[em];B[en];W[ek];B[kq];W[lq];B[lp];W[jq];B[kr];W[jp]
;B[jr];W[mq];B[ip];W[ho];B[gp];W[dq];B[go];W[cp];B[lo];W[kn]
;B[ln];W[km];B[dp];W[hp];B[hq];W[gq];B[gr];W[fq];B[hn];W[jo]
;B[fr];W[do];B[co];W[ep];B[bp];W[md];B[ne];W[of];B[lm];W[le]
;B[kl];W[jl];B[gl];W[kk];B[ll];W[jk];B[mj];W[gj];B[pq];W[pr]
;B[qr];W[po];B[qo];W[on];B[op];W[np];B[pm];W[om];B[no];W[bo]
;B[dp];W[oo];B[pp];W[cp];B[cn];W[nl];B[lk];W[mo];B[pl];W[nj]
;B[ni];W[ok];B[nf];W[mi];B[li];W[mh];B[og];W[pf];B[ki];W[ij]
;B[lg];W[nh];B[oi];W[oh];B[pj];W[ph];B[pk];W[bq];B[dp];W[jg]
;B[kg];W[cp];B[jn];W[in];B[dp];W[fo];B[cp];W[gn];B[ke];W[me]
;B[jf];W[jb];B[ic];W[ib];B[er];W[hc];B[cc];W[bc];B[dd];W[cb]
;B[ce];W[dc];B[cf];W[dg];B[be]
"#, None);

    assert_eq!(board.zobrist_hash(), 6611710067991275858, "wrong hash\n{}", board);
}

/// Test a game featuring a triple ko without the last two
/// moves since they violate the super-ko rules our engine
/// use.
#[test]
fn park_taehee_kim_dayoung() {
    let board = playout_game(r#"
;B[pd];W[dd];B[qp];W[dq];B[co];W[ck];B[ep];W[eq];B[fp];W[fq]
;B[gp];W[hq];B[ci];W[ek];B[cf];W[ce];B[df];W[fd];B[ei];W[en]
;B[cm];W[dm];B[cl];W[dl];B[bk];W[bl];B[bm];W[bj];B[al];W[cj]
;B[gj];W[hl];B[io];W[iq];B[dp];W[cq];B[ik];W[il];B[jl];W[jm]
;B[hk];W[gl];B[im];W[km];B[jn];W[fi];B[fj];W[ej];B[eh];W[hn]
;B[in];W[kn];B[ho];W[jk];B[kl];W[jj];B[ll];W[fh];B[fg];W[gh]
;B[gg];W[hh];B[ii];W[ij];B[hi];W[ih];B[ji];W[hj];B[gi];W[gk]
;B[jh];W[ig];B[ge];W[jg];B[lh];W[kg];B[gd];W[jd];B[cc];W[be]
;B[eb];W[db];B[id];W[jc];B[je];W[ke];B[dc];W[ec];B[cb];W[fb]
;B[da];W[hb];B[gc];W[fc];B[gb];W[ga];B[ib];W[ic];B[hc];W[ha]
;B[fa];W[ea];B[kd];W[db];B[kc];W[jb];B[kb];W[ia];B[ie];W[le]
;B[nc];W[fe];B[gf];W[mc];B[md];W[nd];B[ld];W[me];B[ne];W[kh]
;B[ki];W[li];B[kj];W[lj];B[kk];W[mh];B[ka];W[ja];B[eb];W[fa]
;B[od];W[ch];B[ee];W[db];B[mf];W[lg];B[eb];W[ed];B[bf];W[dh]
;B[bd];W[ae];B[af];W[db];B[di];W[bi];B[eb];W[cd];B[ba];W[bc]
;B[ac];W[ad];B[ab];W[db];B[eg];W[ca];B[dj];W[dk];B[bh];W[ko]
;B[nl];W[pi];B[fn];W[fm];B[da];W[eb];B[dn];W[eo];B[em];W[fk]
;B[pl];W[op];B[ik];W[hk];B[oq];W[om];B[nm];W[on];B[ol];W[pp]
;B[pq];W[qo];B[qq];W[np];B[ro];W[qn];B[rn];W[qm];B[ql];W[rm]
;B[rl];W[mk];B[lm];W[sn];B[ni];W[nh];B[ri];W[qh];B[rh];W[qf]
;B[qg];W[pg];B[rg];W[of];B[mq];W[rp];B[rq];W[mp];B[lq];W[so]
;B[fo];W[el];B[re];W[qe];B[rd];W[bp];B[bo];W[qj];B[rj];W[kr]
;B[nf];W[oj];B[gq];W[gr];B[kq];W[jr];B[qi];W[ph];B[bb];W[sq]
;B[sr];W[rr];B[sp];W[ah];B[bg];W[sq];B[hr];W[fr];B[sp];W[hf]
;B[he];W[sq];B[nk];W[nj];B[sp];W[ml];B[mm];W[sq];B[nq];W[ss]
;B[qr];W[ns];B[ms];W[mr];B[lr];W[ls];B[ks];W[js];B[os];W[ms]
;B[jq];W[ir];B[nr];W[ks];B[lk];W[mj];B[ff];W[rf];B[sf];W[sl]
;B[sk];W[qk];B[rk];W[hm];B[ng];W[mg];B[de];W[sg];B[sh];W[qd]
;B[qc];W[qs];B[lo];W[lp];B[kp];W[ln];B[mn];W[mo];B[jo];W[lo]
;B[ps];W[ip];B[jp];W[ao];B[an];W[ap];B[cp];W[bq];B[pm];W[pn]
;B[nn];W[no];B[og];W[gn];B[oh];W[oi];B[oe];W[pf];B[rs];W[ca]
;B[bd];W[qs];B[da]
"#, None);

    assert_eq!(board.zobrist_hash(), 8830202173104103970, "wrong hash\n{}", board);
}
