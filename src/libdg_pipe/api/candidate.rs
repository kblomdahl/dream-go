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

use dg_go::{Board, Color, Point, utils::{features::{self as features, Features, HWC}, symmetry::Transform}};
use dg_sgf as sgf;
use dg_utils::{b85, types::f16};

use std::ops::Index;

pub struct Candidate {
    board: Board,
    to_move: Color,
    value: f32,
    policy: Option<Vec<f32>>
}

impl Candidate {
    pub fn pass(board: &Board, to_move: Color, value: f32) -> Self {
        Candidate {
            board: board.clone(),
            to_move,
            value,
            policy: Some(pass_policy())
        }
    }

    fn is_pass(&self) -> bool {
        self.policy.as_ref().map(|p| p[361] == 1.0).unwrap_or(false)
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn to_move(&self) -> Color {
        self.to_move
    }

    pub fn value(&self) -> f32 {
        self.value
    }

    pub fn features(&self, symmetry: Transform) -> Vec<f16> {
        features::Default::new(self.to_move(), self.board()).get_features::<HWC, f16>(symmetry)
    }

    pub fn motion_features(&self, symmetry: Transform) -> Vec<f16> {
        features::Default::new(self.to_move(), self.board()).get_motion_features::<HWC, f16>(symmetry)
    }

    pub fn additional_targets(&self, symmetry: Transform) -> Vec<f32> {
        features::Default::new(self.to_move(), self.board()).get_additional_features::<HWC, f32>(symmetry)
    }

    pub fn additional_targets_mask(&self) -> Vec<f32> {
        vec! [1.0; features::Default::num_additional_features()]
    }

    pub fn lz_features(&self, game: &Game, end_index: usize, symmetry: Transform) -> Vec<f16> {
        let mut boards = vec! [];

        for idx in (0..8).filter_map(|i| end_index.checked_sub(i)) {
            boards.push(game[idx].board());
        }

        features::LzFeatures::new(self.to_move(), boards).get_features::<HWC, f16>(symmetry)
    }

    pub fn policy(&self, symmetry: Transform) -> Vec<f32> {
        let mut out = vec! [0.0; 362];
        let original = self.policy.as_ref().unwrap();

        for point in Point::all() {
            let other = symmetry.apply(point);

            out[other.to_packed_index()] = original[point.to_packed_index()];
        }

        out[361] = original[361];
        out
    }
}

pub struct Game {
    candidates: Vec<Candidate>
}

impl Game {
    pub fn from_bytes<'a>(b: &[u8]) -> Option<Self> {
        let mut iter = sgf::Stream::new(b).with_board();
        let mut winner = None;
        let mut out = Self {
            candidates: vec! []
        };

        for (board, tok) in &mut iter {
            match tok {
                sgf::SgfToken::Result { text } if text.len() > 0 => {
                    winner = match text[0] {
                        b'B' => Some(Color::Black),
                        b'W' => Some(Color::White),
                        _ => None
                    };
                },
                sgf::SgfToken::Node { .. } => {
                    out.candidates.push(Candidate {
                        board: board.as_ref().clone(),
                        to_move: Color::Black,
                        value: ::std::f32::NAN,
                        policy: None
                    })
                },
                sgf::SgfToken::Play { .. } => {
                    if let Some(last) = out.candidates.last_mut() {
                        last.to_move = tok.color();
                        last.policy.get_or_insert_with(|| point_to_policy(tok.point()));
                    }
                },
                sgf::SgfToken::Policy { text } => {
                    if let Some(last) = out.candidates.last_mut() {
                        last.policy = b85_to_policy(text);
                    }
                },
                sgf::SgfToken::Value { .. } => {
                    if let Some(last) = out.candidates.last_mut() {
                        last.value = tok.number();
                    }
                }
                _ => { /* pass */ }
            }
        }

        while !iter.is_resign() && out.candidates.len() > 0 && out.pass_count() < 2 {
            out.candidates.push({
                let last = out.candidates.last().unwrap();

                Candidate::pass(
                    iter.board().as_ref().unwrap().as_ref(),
                    last.to_move.opposite(),
                    -last.value
                )
            })
        }

        if out.candidates.is_empty() || winner.is_none() {
            None
        } else {
            out.remove_policyless_candidates();
            out.set_missing_values(winner.unwrap());

            Some(out)
        }
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    fn remove_policyless_candidates(&mut self) {
        self.candidates.retain(|cand| cand.policy.is_some());
    }

    fn set_missing_values(&mut self, winner: Color) {
        for cand in self.candidates.iter_mut() {
            if !cand.value.is_finite() {
                cand.value = if winner == cand.to_move {
                    1.0
                } else {
                    -1.0
                };
            }
        }
    }

    fn pass_count(&self) -> usize {
        self.candidates.iter().rev().take_while(|cand| cand.is_pass()).count()
    }
}

impl Index<usize> for Game {
    type Output = Candidate;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.candidates[idx]
    }
}

/// Returns a policy that represents picking a move at random.
fn pass_policy() -> Vec<f32> {
    let mut out = vec! [0.0; 362];

    out[361] = 1.0;
    out
}

/// Returns a policy that represents a one-hot policy for the given `point`.
fn point_to_policy(point: Point) -> Vec<f32> {
    let mut out = vec! [0.0; 362];

    out[point.to_packed_index()] = 1.0;
    out
}

/// Returns a policy that represents the b85 encoded `f16` policy.
fn b85_to_policy(b: &[u8]) -> Option<Vec<f32>> {
    b85::decode::<f16, f32>(b).filter(|v| v.len() == 362)
}

#[cfg(test)]
mod tests {
    use dg_sgf::{CGoban, ToSgf};

    use super::*;

    #[test]
    fn resigned() {
        let game = Game::from_bytes(b"(;GM[1]FF[4]RE[W+Resign]KM[0.5];B[dd];W[pp])").unwrap();
        let candidates = &game.candidates;

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].board.to_sgf::<CGoban>(), "(;)");
        assert_eq!(candidates[0].to_move, Color::Black);
        assert_eq!(candidates[0].policy, Some(point_to_policy(Point::new(3, 3))));
        assert_eq!(candidates[0].value, -1.0);
        assert_eq!(candidates[1].board.to_sgf::<CGoban>(), "(;AB[dd])");
        assert_eq!(candidates[1].to_move, Color::White);
        assert_eq!(candidates[1].policy, Some(point_to_policy(Point::new(15, 15))));
        assert_eq!(candidates[1].value, 1.0);
    }

    #[test]
    fn scored() {
        let game = Game::from_bytes(b"(;GM[1]FF[4]RE[B+0.5]KM[0.5];B[dd];W[pp])").unwrap();
        let candidates = &game.candidates;

        assert_eq!(candidates.len(), 4);
        assert_eq!(candidates[0].board.to_sgf::<CGoban>(), "(;)");
        assert_eq!(candidates[0].to_move, Color::Black);
        assert_eq!(candidates[0].policy, Some(point_to_policy(Point::new(3, 3))));
        assert_eq!(candidates[0].value, 1.0);
        assert_eq!(candidates[1].board.to_sgf::<CGoban>(), "(;AB[dd])");
        assert_eq!(candidates[1].to_move, Color::White);
        assert_eq!(candidates[1].policy, Some(point_to_policy(Point::new(15, 15))));
        assert_eq!(candidates[1].value, -1.0);
        assert_eq!(candidates[2].board.to_sgf::<CGoban>(), "(;AB[dd]AW[pp])");
        assert_eq!(candidates[2].to_move, Color::Black);
        assert_eq!(candidates[2].policy, Some(pass_policy()));
        assert_eq!(candidates[2].value, 1.0);
        assert_eq!(candidates[3].board.to_sgf::<CGoban>(), "(;AB[dd]AW[pp])");
        assert_eq!(candidates[3].to_move, Color::White);
        assert_eq!(candidates[3].policy, Some(pass_policy()));
        assert_eq!(candidates[3].value, -1.0);
    }
}
