// Copyright 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use dg_utils::config;
use dg_go::{Board, Color, Point};
use dg_sgf as sgf;
use super::{GameResult, Played, predict, greedy_score};
use super::pool::Pool;
use super::predictors::DefaultPredictor;
use super::time_control::RolloutLimit;
use options::StandardSearch;

use crossbeam_channel;
use rand::{Rng, thread_rng};
use std::fs::File;
use std::sync::mpsc;
use std::io::{BufRead, BufReader};
use std::thread;
use std::sync::Arc;

struct Candidate {
    initialized: bool,
    board: Board,
    to_move: Color,
    point: Point
}

struct Game {
    candidates: Vec<Candidate>,
    komi: f32,
    winner: Option<Color>
}

impl Game {
    /// Collect all candidates (moves) from the provided SGF file, and some
    /// game metadata. If an error in encountered while parsing the SGF file
    /// a partial set of candidates may be provided.
    ///
    /// # Arguments
    ///
    /// * `content` -
    ///
    fn from_str(content: &str) -> Game {
        let mut iter = sgf::Stream::new(content.as_bytes()).with_board();
        let mut out = Self {
            candidates: vec! [],
            komi: 0.5,
            winner: None
        };

        for (board, tok) in &mut iter {
            eprintln!("{:?}", tok);

            match tok {
                sgf::SgfToken::Result { text } if text.len() > 0 => {
                    out.winner = match text[0] {
                        b'B' => Some(Color::Black),
                        b'W' => Some(Color::White),
                        _ => None
                    };
                },
                sgf::SgfToken::Node { .. } => {
                    out.candidates.push(Candidate {
                        initialized: false,
                        board: board.as_ref().clone(),
                        to_move: Color::Black,
                        point: Point::default()
                    })
                },
                sgf::SgfToken::Play { .. } => {
                    if let Some(last) = out.candidates.last_mut() {
                        last.initialized = true;
                        last.to_move = tok.color();
                        last.point = tok.point();
                    }
                },
                _ => { /* pass */ }
            }
        }

        out.komi = iter.komi();
        while iter.is_resign() && out.candidates.len() > 0 && out.pass_count() < 2 {
            out.candidates.push({
                let last = out.candidates.last().unwrap();

                Candidate {
                    initialized: true,
                    board: last.board.clone(),
                    to_move: last.to_move.opposite(),
                    point: Point::default()
                }
            })
        }

        out.candidates.retain(|cand| cand.initialized);
        out
    }

    fn pass_count(&self) -> usize {
        self.candidates.iter().rev().take_while(|cand| cand.point == Point::default()).count()
    }
}

/// Reanalyze a given `candidate`.
///
/// # Arguments
///
/// * `server` -
/// * `candidate` -
///
fn reanalyze_single_candidate(
    pool: &Pool,
    candidate: &Candidate
) -> Option<Played>
{
    let result = predict(
        pool,
        Box::new(StandardSearch::new()),
        Box::new(RolloutLimit::new(usize::from(*config::NUM_ROLLOUT))),
        None,
        &candidate.board,
        candidate.to_move
    );

    result.map(|(value, _, tree)| {
        Played::from_mcts(candidate.to_move, candidate.point, value, &tree)
    })
}

/// If the provided `candidate` is a good candidate for reanalyzing then
/// return `Some(candiate)`, otherwise `None` (for chaining purposes).
///
/// # Arguments
///
/// * `candidate` -
///
fn is_good_candidate(candidate: &Candidate) -> Option<&Candidate> {
    if thread_rng().gen::<f32>() < 0.05 {
        Some(candidate)
    } else {
        None
    }
}

/// Run the re-analyze proceedure on the provided SGF file and return
/// a game result with the re-analyzed results embedded.
///
/// # Arguments
///
/// * `server` -
/// * `content` -
///
fn reanalyze_single_line(
    pool: &Pool,
    content: String
) -> Option<GameResult>
{
    let game = Game::from_str(&content);
    let mut board = Board::new(game.komi);
    let mut sgf = String::new();

    for cand in &game.candidates {
        let analyze = is_good_candidate(cand);

        if let Some(played) = analyze.and_then(|cand| reanalyze_single_candidate(pool, cand)) {
            sgf += &format!("{}", played);
        } else {
            sgf += &format!("{}", Played::fixed(cand.to_move, cand.point));
        }

        assert!(board.is_valid(cand.to_move, cand.point));
        board.place(cand.to_move, cand.point);
    }

    if game.winner.is_none() {
        let last_played = game.candidates.last().map(|cand| cand.to_move.opposite());

        if let Some(to_move) = last_played {
            let (greedy_board, _) = greedy_score(pool.predictor(), &board, to_move);

            Some(GameResult::Ended(sgf, greedy_board))
        } else {
            None
        }
    } else if let Some(re) = game.winner {
        Some(GameResult::Resign(sgf, board, re, 0.5))
    } else {
        None
    }
}

/// Read each line in the provided file, and send it over the provided `channel`.
///
/// # Arguments
///
/// * `file` -
/// * `sender` -
///
fn parse_single_file(file: String, sender: crossbeam_channel::Sender<String>) {
    if let Ok(f) = File::open(file) {
        for line in BufReader::new(&f).lines() {
            if let Ok(line) = line {
                if sender.send(line).is_err() {
                    break;
                }
            }
        }
    }
}

/// For each input file spawn a thread that reads the lines and put
/// them on the lines channel.
///
/// # Arguments
///
/// * `files` -
///
fn spawn_file_workers(files: &[String]) -> crossbeam_channel::Receiver<String> {
    let (sender, receiver) = crossbeam_channel::bounded(5);

    for file in files.iter().cloned() {
        let sender = sender.clone();

        thread::spawn(move || parse_single_file(file, sender));
    }

    receiver
}

pub fn reanalyze(
    files: &[String]
) -> (mpsc::Receiver<GameResult>, Arc<Pool>)
{
    let pool = Arc::new(Pool::new(Box::new(DefaultPredictor::default())));
    let lines = spawn_file_workers(files);

    // spawn the worker threads that generate the self-play games
    let num_parallel = config::NUM_GAMES.max(1);
    let (sender, receiver) = mpsc::channel();

    for _ in 0..num_parallel {
        let sender = sender.clone();
        let lines = lines.clone();
        let pool = pool.clone();

        thread::spawn(move || {
            for line in &lines {
                if let Some(result) = reanalyze_single_line(pool.as_ref(), line) {
                    sender.send(result).unwrap();
                }
            }
        });
    }

    (receiver, pool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dg_sgf::{ToSgf, CGoban};

    #[test]
    fn collect_all_candidates() {
        let game = Game::from_str(&"(;GM[1]FF[4]SZ[19];B[aa];W[bb];B[cc];W[dd])");
        let actual = &game.candidates;

        assert_eq!(actual.len(), 4);
        assert_eq!(actual[0].board.to_sgf::<CGoban>(), "()");
        assert_eq!(actual[0].to_move, Color::Black);
        assert_eq!(actual[0].point, Point::new(0, 0));
        assert_eq!(actual[1].board.to_sgf::<CGoban>(), "(AB[aa])");
        assert_eq!(actual[1].to_move, Color::White);
        assert_eq!(actual[1].point, Point::new(1, 1));
        assert_eq!(actual[2].board.to_sgf::<CGoban>(), "(AB[aa]AW[bb])");
        assert_eq!(actual[2].to_move, Color::Black);
        assert_eq!(actual[2].point, Point::new(2, 2));
        assert_eq!(actual[3].board.to_sgf::<CGoban>(), "(AB[aa][cc]AW[bb])");
        assert_eq!(actual[3].to_move, Color::White);
        assert_eq!(actual[3].point, Point::new(3, 3));
    }
}
