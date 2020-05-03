// Copyright 2020 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
use super::choose::choose;
use super::predict::Predictor;
use super::time_control::{TimeStrategy, RolloutLimit};
use super::{GameResult, get_random_komi};
use super::{predict_service, predict_aux, full_forward, tree};
use dg_nn::Network;
use options::{SearchOptions, StandardSearch, ScoringSearch};

use rand::{Rng, thread_rng};
use std::fmt::{self, Display, Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::thread;
use ordered_float::OrderedFloat;

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
    num_rollout: usize,
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
            num_rollout: 0,
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
        let num_rollout = tree.size();

        Self {
            to_move,
            point,
            value,
            num_rollout,
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
        let num_rollout = 1;

        Self {
            to_move,
            point,
            value,
            num_rollout,
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

        if self.num_rollout <= 1 {
            Ok(())
        } else {
            write!(
                f,
                "TV[{}]P[{}]V[{:.4}]",
                self.num_rollout,
                b85::encode(&self.softmax),
                if self.to_move == Color::Black {
                    2.0 * self.value - 1.0
                } else {
                    -2.0 * self.value + 1.0
                }
            )
        }
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
    /// will be a value between `*config::NUM_ROLLOUT` and 10% of it.
    fn num_rollout(&self) -> usize {
        let max_rollout: usize = (*config::NUM_ROLLOUT).into();
        let winrate = self.winrate.get();
        let m = 4.0 * winrate * (1.0 - winrate);
        let m = if m < 0.1 { 0.1 } else { m };

        (m * (max_rollout as f32)) as usize
    }

    fn predict_aux<P: Predictor + 'static, T: TimeStrategy + Clone + Send + 'static>(
        &mut self,
        board: &Board,
        allow_pass: bool,
        server: &P,
        num_workers: usize,
        time_control: T
    ) -> Option<(f32, usize, tree::Node<O>)>
    {
        if !allow_pass {
            let (value, index, tree) = predict_aux::<_, _, ScoringSearch>(
                server,
                num_workers,
                time_control,
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
                time_control,
                self.root.take(),
                &board,
                self.color
            )
        }
    }

    /// Perform an expert iteration, replacing the stored search tree, but not
    /// sugggested move, in the played move.
    /// 
    /// # Arguments
    /// 
    /// * `board` -
    /// * `point` - 
    /// * `allow_pass` -
    /// * `server` -
    /// * `num_workers` -
    /// 
    fn ex_it<P: Predictor + 'static>(
        &mut self,
        board: &Board,
        point: Point,
        allow_pass: bool,
        server: &P,
        num_workers: usize
    ) -> Option<Played>
    {
        let (value, _, tree) = self.predict_aux(
            board,
            allow_pass,
            server,
            num_workers,
            RolloutLimit::new((*config::NUM_EX_IT_ROLLOUT).into())
        )?;

        debug_assert!(0.0 <= value && value <= 1.0, "{}", value);

        Some(Played::from_mcts(self.color, point, value, &tree))
    }

    /// Predict the best next move for the given board state. If `ex_it` is
    /// given then we will replace the stored policy with a full search tree.
    /// 
    /// # Arguments
    /// 
    /// * `board` -
    /// * `allow_pass`  whether we are allowed to pass
    /// * `ex_it` - 
    /// * `server` - 
    /// * `num_workers` -
    /// 
    fn predict<P: Predictor + 'static>(
        &mut self,
        board: &Board,
        allow_pass: bool,
        ex_it: bool,
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
                RolloutLimit::new(num_rollout)
            )?;

            if !value.is_finite() {
                self.root = None;
                return Some(Played::pass(self.color));
            }

            debug_assert!(0.0 <= value && value <= 1.0, "{}", value);
            debug_assert!(index < 362, "{}", index);

            // update internal state
            let point = Point::from_packed_parts(index);
            let played =
                if ex_it {
                    self.ex_it(board, point, allow_pass, server, num_workers)?
                } else {
                    Played::from_mcts(self.color, point, value, &tree)
                };

            self.winrate.update(value);
            self.root = tree::Node::forward(tree, index);

            Some(played)
        } else {
            let (value, mut policy) =
                if allow_pass {
                    full_forward::<_, O>(server, board, self.color)?
                } else {
                    full_forward::<_, ScoringSearch>(server, board, self.color)?
                };
            if !allow_pass {
                policy[361] = ::std::f32::NEG_INFINITY;
            }

            let index = choose(
                &policy.iter().map(|&x| OrderedFloat(x as f64)).collect::<Vec<_>>(),
                0.5,
                1.0 / *config::TEMPERATURE as f64,
                thread_rng().gen::<f64>()
            ).map(|(i, _)| i).unwrap_or(361);

            debug_assert!(0.0 <= value && value <= 1.0, "{}", value);
            debug_assert!(index < 362, "{}", index);

            // update internal state
            let point = Point::from_packed_parts(index);
            let played =
                if ex_it {
                    self.ex_it(board, point, allow_pass, server, num_workers)?
                } else {
                    Played::from_forward(self.color, point, value, policy)
                };

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
/// * `ex_it` - whether to enable with expert iteration
///
fn self_play_one<P: Predictor + 'static>(
    server: &P,
    num_parallel: &Arc<AtomicUsize>,
    ex_it: bool
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

        let allow_pass = board.is_scorable();
        let ex_it = ex_it && thread_rng().gen::<f32>() < 0.01;
        let played = players[0].predict(&mut board, allow_pass, ex_it, server, num_workers)?;
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
/// * `ex_it` - whether to enable with expert iteration
///
pub fn self_play(
    network: Network,
    num_games: usize,
    ex_it: bool
) -> (Receiver<GameResult>, predict_service::PredictService)
{
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
                if let Some(result) = self_play_one(&server, &num_workers, ex_it) {
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
