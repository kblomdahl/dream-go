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

use dg_go::utils::score::{Score, StoneStatus};
use dg_go::utils::sgf::{CGoban, SgfCoordinate};
use dg_go::{Board, Color};

use std::fmt;

pub enum GameResult {
    Resign(String, Board, Color, f32),
    Ended(String, Board)
}

impl fmt::Display for GameResult {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let now = time::now_utc();
        let iso8601 = time::strftime("%Y-%m-%dT%H:%M:%S%z", &now).unwrap();

        match *self {
            GameResult::Resign(ref sgf, ref board, winner, _) => {
                write!(fmt, "(;GM[1]FF[4]DT[{}]SZ[19]RU[Chinese]KM[{:.1}]RE[{}+Resign]{})", iso8601, board.komi(), winner, sgf)
            },
            GameResult::Ended(ref sgf, ref board) => {
                let winner = get_winner_as_sgf(board);
                let territory = get_territory_as_sgf(board);

                write!(fmt, "(;GM[1]FF[4]DT[{}]SZ[19]RU[Chinese]KM[{:.1}]RE[{}]{}{})", iso8601, board.komi(), winner, sgf, territory)
            }
        }
    }
}

/// Returns the territory for both colors of the given board, according to
/// TT-rules, as `TB` and `TW` properties.
/// 
/// # Arguments
/// 
/// * `board` - 
/// 
fn get_territory_as_sgf(board: &Board) -> String {
    let mut black = String::new();
    let mut white = String::new();

    for (point, statuses) in board.get_stone_status(&board) {
        if statuses.contains(&StoneStatus::WhiteTerritory) {
            white += &format!("[{}]", CGoban::to_sgf(point));
        } else if statuses.contains(&StoneStatus::BlackTerritory) {
            black += &format!("[{}]", CGoban::to_sgf(point));
        }
    }

    format!(
        "{}{}{}{}",
        if black.len() > 0 { "TB" } else { "" },
        black,
        if white.len() > 0 { "TW" } else { "" },
        white
    )
}

/// Returns the winner of the given board, according to TT-rules, as an SGF
/// property.
/// 
/// # Arguments
/// 
/// * `board` - 
/// 
fn get_winner_as_sgf(board: &Board) -> String {
    let (black, white) = board.get_score();
    let black = black as f32;
    let white = white as f32 + board.komi();

    if black > white {
        format!("B+{:.1}", black - white)
    } else if white > black {
        format!("W+{:.1}", white - black)
    } else {
        "0".to_string()
    }
}
