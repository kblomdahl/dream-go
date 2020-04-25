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
//

use dg_go::{DEFAULT_KOMI, Board, Color, Point};
use dg_nn;

use regex::Regex;
use std::fs::File;
use std::io::prelude::*;

thread_local! {
    #[allow(dead_code)]
    static NETWORK: dg_nn::Network = dg_nn::Network::new().unwrap();
}

/// Play each move in the given SGF string and return the final board state,
/// if any of the moves are invalid then it panic.
///
/// # Arguments
///
/// * `src` - the SGF string
/// * `max_moves` -
///
#[allow(dead_code)]
pub fn playout_game(src: &str, max_moves: Option<usize>) -> Board {
    lazy_static! {
        static ref LETTERS: [char; 26] = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z'
        ];
        static ref MOVE: Regex = Regex::new(r";([BW])\[([a-z]*)\]").unwrap();
    }

    let mut board = Board::new(DEFAULT_KOMI);
    let mut count = 1;

    for cap in MOVE.captures_iter(src) {
        let color = match &cap[1] {
            "B" => Color::Black,
            "W" => Color::White,
            _   => { unreachable!(); }
        };
        let x = cap[2].chars().nth(0)
            .and_then(|x| LETTERS.binary_search(&x).ok())
            .unwrap_or(board.size());
        let y = cap[2].chars().nth(1)
            .and_then(|y| LETTERS.binary_search(&y).ok())
            .unwrap_or(board.size());

        if x < board.size() && y < board.size() {
            let point = Point::new(x, 18 - y);
            assert!(board.is_valid(color, point), "invalid move {}: {} {} {}\n{}", count, color, x, y, board);

            board.place(color, point);
        }
        count += 1;

        if max_moves.map(|m| count >= m).unwrap_or(false) {
            break
        }
    }

    board
}

/// Loads the content of the given SGF file and return the board state after
/// at most `max_moves` moves has been played. If the file contains an invalid
/// SGF file, or invalid moves then this method will panic.
/// 
/// # Argumnets
/// 
/// * `filename` -
/// * `max_moves` -
/// 
#[allow(dead_code)]
pub fn playout_file(filename: &str, max_moves: Option<usize>) -> Board {
    let mut file = File::open(filename).expect("file not found");
    let mut contents = String::new();

    file.read_to_string(&mut contents).expect("something went wrong reading the file");

    playout_game(&contents, max_moves)
}

