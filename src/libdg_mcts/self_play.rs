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

use dg_go::utils::score::Score;
use dg_go::utils::sgf::{CGoban, SgfCoordinate};
use dg_go::{Board, Color, Point};
use dg_utils::{b85, config};
use super::asm::argmax_f32;
use super::predict::Predictor;
use super::time_control::RolloutLimit;
use super::{GameResult, get_random_komi};
use super::{predict_service, predict_aux, full_forward, tree};
use dg_nn::Network;
use options::{SearchOptions, StandardSearch, ScoringSearch};

use std::fmt::{self, Display, Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::thread;

/// The momentum to use when updating the moving average of the winrate.
const MOMENTUM: f32 = 0.2;

/// An moving average of values.
struct MovingAverage {
    average: f32,
    momentum: f32,
}

impl MovingAverage {
    fn new(initial_value: f32, momentum: f32) -> Self {
        Self {
            average: initial_value,
            momentum: momentum,
        }
    }

    fn get(&self) -> f32 {
        self.average
    }

    fn update(&mut self, next_value: f32) {
        self.average -= self.momentum * (self.average - next_value);
    }
}

/// A move that has been played in the game, together with the meta-data about
/// why we're playing this move.
struct Played {
    to_move: Color,
    point: Point,
    value: f32,
    explain: String,
    softmax: Vec<f32>,
    prior_point: Point,
}

impl Played {
    fn pass(to_move: Color) -> Self {
        Self {
            to_move: to_move,
            point: Point::default(),
            value: 0.0,
            explain: String::new(),
            softmax: vec! [],
            prior_point: Point::default()
        }
    }

    fn from_mcts<O: SearchOptions + 'static>(
        to_move: Color,
        point: Point,
        value: f32,
        tree: &tree::Node<O>
    ) -> Self
    {
        let (_, prior_index) = tree.prior();
        let prior_point = Point::from_packed_parts(prior_index);
        let softmax = tree.softmax();
        let explain = tree::to_pretty(tree).to_string();

        Self {
            to_move,
            point,
            value,
            explain,
            softmax,
            prior_point,
        }
    }

    fn from_forward(
        to_move: Color,
        point: Point,
        value: f32,
        softmax: Vec<f32>,
    ) -> Self
    {
        let prior_point = Point::default();
        let explain = String::new();

        Self {
            to_move,
            point,
            value,
            explain,
            softmax,
            prior_point,
        }
    }
}

impl Display for Played {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, ";{}[{}]", self.to_move, CGoban::to_sgf(self.point))?;

        if !self.explain.is_empty() {
            write!(f, "C[{}]", self.explain.replace("\n", "\r"))?;
        }

        if self.prior_point != Point::default() {
            write!(f, "TR[{}]", CGoban::to_sgf(self.prior_point))?;
        }

        write!(
            f,
            "P[{}]V[{:.4}]",
            b85::encode(&self.softmax),
            if self.to_move == Color::Black {
                2.0 * self.value - 1.0
            } else {
                -2.0 * self.value + 1.0
            }
        )
    }
}

/// An AI-player in a game.
struct Player<O: SearchOptions + 'static> {
    winrate: MovingAverage,
    root: Option<tree::Node<O>>,
    color: Color,
}

impl<O: SearchOptions + 'static> Player<O> {
    fn new(color: Color) -> Self {
        Self {
            winrate: MovingAverage::new(0.5, MOMENTUM),
            root: None,
            color: color,
        }
    }

    /// Returns the number of rollouts to perform for the current winrate. This
    /// will be a value between `*config::NUM_ROLLOUT` and 100.
    fn num_rollout(&self) -> usize {
        let max_rollout: usize = (*config::NUM_ROLLOUT).into();
        let winrate = self.winrate.get();

        ::std::cmp::max(
            100,
            (4.0 * winrate * (1.0 - winrate) * (max_rollout as f32)) as usize
        )
    }

    fn predict_aux<P: Predictor + 'static>(
        &mut self,
        board: &Board,
        allow_pass: bool,
        server: &P,
        num_workers: usize,
        num_rollout: usize
    ) -> Option<(f32, usize, tree::Node<O>)>
    {
        if !allow_pass {
            let (value, index, tree) = predict_aux::<_, _, ScoringSearch>(
                server,
                num_workers,
                RolloutLimit::new(num_rollout),
                self.root.take().map(|mut n| {
                    n.disqualify(361);
                    n.to_options::<ScoringSearch>()
                }),
                &board,
                self.color
            )?;

            Some((value, index, tree.to_options::<O>()))
        } else {
            predict_aux::<_, _, O>(
                server,
                num_workers,
                RolloutLimit::new(num_rollout),
                self.root.take(),
                &board,
                self.color
            )
        }
    }

    /// Predict a single move 
    fn predict<P: Predictor + 'static>(
        &mut self,
        board: &Board,
        allow_pass: bool,
        server: &P,
        num_workers: usize
    ) -> Option<Played>
    {
        let num_rollout = self.num_rollout();

        if num_rollout > 1 {
            let (value, index, tree) = self.predict_aux(
                board,
                allow_pass,
                server,
                num_workers,
                num_rollout
            )?;

            if !value.is_finite() {
                self.root = None;
                return Some(Played::pass(self.color));
            }

            debug_assert!(0.0 <= value && value <= 1.0, "{}", value);
            debug_assert!(index < 362, "{}", index);

            // update internal state
            let point = Point::from_packed_parts(index);
            let played = Played::from_mcts(self.color, point, value, &tree);

            self.winrate.update(value);
            self.root = tree::Node::forward(tree, index);

            Some(played)
        } else {
            let (value, mut policy) = full_forward::<_, O>(server, board, self.color)?;
            if !allow_pass {
                policy[361] = ::std::f32::NEG_INFINITY;
            }

            let index = argmax_f32(&policy).unwrap_or(361);

            debug_assert!(0.0 <= value && value <= 1.0, "{}", value);
            debug_assert!(index < 362, "{}", index);

            // update internal state
            let point = Point::from_packed_parts(index);
            let played = Played::from_forward(self.color, point, value, policy);

            self.winrate.update(value);
            self.forward(point);

            Some(played)
        }
    }

    fn forward(&mut self, point: Point) {
        if let Some(tree) = self.root.take() {
            self.root = tree::Node::forward(tree, point.to_packed_index());
        }
    }
}

/// Play a game against the engine and return the result of the game.
///
/// # Arguments
///
/// * `server` - the server to use during evaluation
/// * `num_parallel` - the number of games that are being played in parallel
///
fn self_play_one<P: Predictor + 'static>(
    server: &P,
    num_parallel: &Arc<AtomicUsize>
) -> Option<GameResult>
{
    let mut board = Board::new(get_random_komi());
    let mut sgf = String::new();
    let mut pass_count = 0;

    let mut players: Vec<Player<StandardSearch>> = vec! [
        Player::new(Color::Black),
        Player::new(Color::White)
    ];

    while board.count() < 722 {
        let num_workers =
            ::std::cmp::max(
                1,
                *config::NUM_THREADS / num_parallel.load(Ordering::Acquire)
            );

        let allow_pass = pass_count < 2 || board.is_scorable();
        let played = players[0].predict(&mut board, allow_pass, server, num_workers)?;
        sgf += &format!("{}", played);

        if played.point == Point::default() {  // passing move
            pass_count += 1;

            if pass_count >= 2 && board.is_scorable() {
                return Some(GameResult::Ended(sgf, board))
            }
        } else {
            pass_count = 0;
            board.place(players[0].color, played.point);
        }

        // swap whose turn it is to place a stone
        players[1].forward(played.point);
        players.reverse();
    }

    Some(GameResult::Ended(sgf, board))
}

/// Play games against the engine and return the result of the games
/// over the channel.
///
/// # Arguments
///
/// * `network` - the neural network to use during evaluation
/// * `num_games` - the number of games to generate
///
pub fn self_play(network: Network, num_games: usize) -> (Receiver<GameResult>, predict_service::PredictService) {
    let server = predict_service::service(network);
    let (sender, receiver) = channel();

    // spawn the worker threads that generate the self-play games
    let num_parallel = ::std::cmp::min(num_games, *config::NUM_GAMES);
    let num_workers = Arc::new(AtomicUsize::new(num_parallel));
    let processed = Arc::new(AtomicUsize::new(0));

    for _ in 0..num_parallel {
        let num_workers = num_workers.clone();
        let processed = processed.clone();
        let sender = sender.clone();
        let server = server.lock().clone_to_static();

        thread::spawn(move || {
            while processed.fetch_add(1, Ordering::SeqCst) < num_games {
                if let Some(result) = self_play_one(&server, &num_workers) {
                    eprint!(".");
                    if sender.send(result).is_err() {
                        break
                    }
                }
            }

            num_workers.fetch_sub(1, Ordering::Release);
        });
    }

    (receiver, server)
}
