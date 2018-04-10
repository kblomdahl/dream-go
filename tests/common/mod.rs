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

use dream_go::go::symmetry::Transform;
use dream_go::go::{Board, Color, Features, Score, CHW, HWC};
use dream_go::mcts;
use dream_go::nn;
use dream_go::util::types::*;

use regex::Regex;
use std::fs::File;
use std::io::prelude::*;

thread_local! {
    #[allow(dead_code)]
    static NETWORK: nn::Network = nn::Network::new().unwrap();
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

    let mut board = Board::new();
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
            assert!(board.is_valid(color, x, 18 - y), "invalid move {}: {} {} {}\n{}", count, color, x, y, board);

            board.place(color, x, 18 - y);
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

/// Returns the next move as suggested by the neural network for the given
/// color.
/// 
/// # Arguments
/// 
/// * `board` -
/// * `next_color` -
/// 
#[allow(dead_code)]
pub fn predict(board: &Board, next_color: Color) -> (f32, Box<[f32]>) {
    // predict the next move and the current value using the neural network
    NETWORK.with(|network| {
        let mut workspace = network.get_workspace(1);

        match *nn::TYPE {
            nn::Type::Single => {
                let features = board.get_features::<f32, CHW>(next_color, Transform::Identity);
                let (value, policy) = nn::forward::<f32, f32>(&mut workspace, &vec! [features]);

                (value[0], policy[0].clone())
            },
            nn::Type::Half => {
                let features = board.get_features::<f16, CHW>(next_color, Transform::Identity);
                let (value, policy) = nn::forward::<f16, f16>(&mut workspace, &vec! [features]);
                let policy = policy[0].iter()
                    .map(|&p| f32::from(p))
                    .collect::<Vec<f32>>();

                (f32::from(value[0]), policy.into_boxed_slice())
            },
            nn::Type::Int8 => {
                let features = board.get_features::<q8, HWC>(next_color, Transform::Identity);
                let (value, policy) = nn::forward::<q8, f32>(&mut workspace, &vec! [features]);

                (value[0], policy[0].clone())
            }
        }
    })
}

/// Returns the final score as suggested by the neural network for the given
/// next color to play.
/// 
/// # Arguments
/// 
/// * `board` -
/// * `next_color` -
/// 
#[allow(dead_code)]
pub fn greedy_score(board: &Board, next_color: Color) -> (usize, usize) {
    NETWORK.with(|network| {
        let service = mcts::predict::service(network.clone());
        let (finished, _rollout) = mcts::greedy_score(
            &service.lock(),
            board,
            next_color
        );

        board.get_guess_score(&finished)
    })
}
