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
use dg_go::{Board, Color, Point};
use dg_sgf::{CGoban, ToSgf};
use dg_utils::{b85, config};
use super::{predict, full_forward, tree, GameResult, get_random_komi};
use super::asm::sum_finite_f32;
use super::choose::choose;
use super::pool::Pool;
use super::predictors::DefaultPredictor;
use super::time_control::{TimeStrategy, RolloutLimit};
use options::{SearchOptions, StandardSearch, ScoringSearch};

use rand::{Rng, thread_rng};
use std::fmt::{self, Display, Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
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

/// Returns the skewness of the given vector as defined by Pearson's moment
/// coefficient of skewness [1].
///
/// [1] https://en.wikipedia.org/wiki/Skewness
///
/// # Arguments
///
/// * `values` -
///
#[allow(dead_code)]
fn skewness(values: &[f32]) -> f32 {
    let recip_len = (values.len() as f32).recip();
    let mean = sum_finite_f32(values) * recip_len;
    let mut k_2 = 0.0;
    let mut k_3 = 0.0;

    for &x in values.iter() {
        if x.is_finite() {
            let delta = x - mean;

            k_2 += delta.powi(2) * recip_len;
            k_3 += delta.powi(3) * recip_len;
        }
    }

    k_3 / k_2.powf(1.5)
}

/// A move that has been played in the game, together with the meta-data about
/// why we're playing this move.
pub struct Played {
    to_move: Color,
    point: Point,
    value: Option<f32>,
    num_rollout: usize,
    explain: String,
    softmax: Vec<f32>,
    prior_point: Point,
}

impl Played {
    pub fn fixed(to_move: Color, point: Point) -> Self {
        Self {
            to_move: to_move,
            point: point,
            value: None,
            num_rollout: 0,
            explain: String::new(),
            softmax: vec! [],
            prior_point: Point::default()
        }
    }

    pub fn pass(to_move: Color) -> Self {
        Self {
            to_move: to_move,
            point: Point::default(),
            value: None,
            num_rollout: 0,
            explain: String::new(),
            softmax: vec! [],
            prior_point: Point::default()
        }
    }

    pub fn from_mcts(
        to_move: Color,
        point: Point,
        value: f32,
        tree: &tree::Node
    ) -> Self
    {
        let (_, prior_index) = tree.prior();
        let prior_point = Point::from_packed_parts(prior_index);
        let softmax = tree.softmax();
        let explain = tree::to_pretty(tree).to_string();
        let num_rollout = tree.size();
        let value = Some(value);

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

    pub fn from_forward(
        to_move: Color,
        point: Point,
        value: f32,
        softmax: Vec<f32>,
    ) -> Self
    {
        let prior_point = Point::default();
        let explain = String::new();
        let num_rollout = 1;
        let value = Some(value);

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

    /// Returns a normalized win rate that always refects the probability
    /// that black will win.
    fn normalized_win_rate(&self) -> Option<f32> {
        self.value.map(|value| {
            if self.to_move == Color::Black {
                2.0 * value - 1.0
            } else {
                -2.0 * value + 1.0
            }
        })
    }
}

impl Display for Played {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, ";{}[{}]", self.to_move, self.point.to_sgf::<CGoban>())?;

        if !self.explain.is_empty() {
            write!(f, "C[{}]", self.explain.replace("\n", "\r"))?;
        }

        if self.prior_point != Point::default() {
            write!(f, "TR[{}]", self.prior_point.to_sgf::<CGoban>())?;
        }

        if self.num_rollout > 1 {
            write!(
                f,
                "TV[{}]P[{}]",
                self.num_rollout,
                b85::encode(&self.softmax)
            )?;
        }

        if let Some(value) = self.normalized_win_rate() {
            write!(f, "V[{:.4}]", value)
        } else {
            Ok(())
        }
    }
}

/// An AI-player in a game.
struct Player {
    winrate: MovingAverage,
    root: Option<tree::Node>,
    color: Color,
}

impl Player {
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

    fn predict_aux(
        &mut self,
        board: &Board,
        allow_pass: bool,
        pool: &Pool,
        time_strategy: Box<dyn TimeStrategy + Sync>
    ) -> Option<(f32, usize, tree::Node)>
    {
        if !allow_pass {
            let (value, index, tree) = predict(
                pool,
                Box::new(ScoringSearch::new()),
                time_strategy,
                self.root.take().map(|mut n| {
                    n.disqualify(361);
                    n
                }),
                &board,
                self.color
            )?;

            Some((value, index, tree))
        } else {
            predict(
                pool,
                Box::new(StandardSearch::new()),
                time_strategy,
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
    ///
    fn ex_it(
        &mut self,
        board: &Board,
        point: Point,
        allow_pass: bool,
        pool: &Pool,
    ) -> Option<Played>
    {
        let (value, _, tree) = self.predict_aux(
            board,
            allow_pass,
            pool,
            Box::new(RolloutLimit::new((*config::NUM_EX_IT_ROLLOUT).into()))
        )?;

        debug_assert!(0.0 <= value && value <= 1.0, "{}", value);

        Some(Played::from_mcts(self.color, point, value, &tree))
    }

    /// Returns true if the given skewness of the policy indicates that this
    /// move is a good candidate for policy extraction.
    ///
    /// # Arguments
    ///
    /// * `value` -
    /// * `policy` -
    ///
    fn is_good_candidate(&self, value: f32, _policy: &[f32]) -> bool {
        value >= -0.80 && value <= 0.80 && {
            thread_rng().gen::<f32>() < 0.05
        }
    }

    /// Predict the best next move for the given board state. If `ex_it` is
    /// given then we will replace the stored policy with a full search tree.
    ///
    /// # Arguments
    ///
    /// * `board` -
    /// * `allow_pass`  whether we are allowed to pass
    /// * `ex_it` -
    /// * `pool` -
    ///
    fn predict(
        &mut self,
        board: &Board,
        allow_pass: bool,
        ex_it: bool,
        pool: &Pool,
    ) -> Option<Played>
    {
        let num_rollout = self.num_rollout();

        if num_rollout > 1 {
            let (value, index, tree) = self.predict_aux(
                board,
                allow_pass,
                pool,
                Box::new(RolloutLimit::new(num_rollout))
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
                if ex_it && self.is_good_candidate(value, &tree.softmax()) {
                    self.ex_it(board, point, allow_pass, pool)?
                } else {
                    Played::from_mcts(self.color, point, value, &tree)
                };

            self.winrate.update(value);
            self.root = tree::Node::forward(tree, index);

            Some(played)
        } else {
            let search_options: Box<dyn SearchOptions + Sync> =
                if allow_pass {
                    Box::new(StandardSearch::default())
                } else {
                    Box::new(ScoringSearch::default())
                };
            let (value, mut policy, _hidden_states) = full_forward(pool.predictor(), &search_options, board, self.color)?;
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
                if ex_it && self.is_good_candidate(value, &policy) {
                    self.ex_it(board, point, allow_pass, pool)?
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
/// * `pool` - the pool to use during evaluation
/// * `num_parallel` - the number of games that are being played in parallel
/// * `ex_it` - whether to enable with expert iteration
///
fn self_play_one(
    pool: &Pool,
    ex_it: bool
) -> Option<GameResult>
{
    let mut board = Board::new(get_random_komi());
    let mut sgf = String::new();
    let mut pass_count = 0;

    let mut players: Vec<Player> = vec! [
        Player::new(Color::Black),
        Player::new(Color::White)
    ];

    while board.count() < 722 {
        let allow_pass = board.is_scorable();
        let played = players[0].predict(&mut board, allow_pass, ex_it, pool)?;
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
    num_games: usize,
    ex_it: bool
) -> (Receiver<GameResult>, Arc<Pool>)
{
    let pool = Arc::new(Pool::new(Box::new(DefaultPredictor::default())));

    // spawn the worker threads that generate the self-play games
    let num_parallel = num_games.min(*config::NUM_GAMES);
    let (sender, receiver) = sync_channel(3 * num_parallel);
    let processed = Arc::new(AtomicUsize::new(0));

    for _ in 0..num_parallel {
        let processed = processed.clone();
        let sender = sender.clone();
        let pool = pool.clone();

        thread::spawn(move || {
            while processed.fetch_add(1, Ordering::AcqRel) < num_games {
                if let Some(result) = self_play_one(pool.as_ref(), ex_it) {
                    if sender.send(result).is_err() {
                        break
                    }
                }
            }
        });
    }

    (receiver, pool)
}

#[cfg(test)]
mod tests {
    use ::options::StandardDeterministicSearch;
    use ::predictors::FakePredictor;
    use super::*;

    #[test]
    fn moving_average() {
        let mut avg = MovingAverage::new(0.5, 0.2);
        let mut prev_distance = 0.5;

        for _ in 0..10 {
            avg.update(1.0);
            let distance = 1.0 - avg.get();

            assert!(distance < prev_distance, "{} < {}", distance, prev_distance);
            prev_distance = distance;
        }
    }

    #[test]
    fn normal_skewness() {
        let values = vec! [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];

        assert_eq!(skewness(&values), 0.0);
    }

    #[test]
    fn positive_skewness() {
        let values = vec! [-4.0, -3.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0];

        assert!(skewness(&values) > 1e-3);
    }

    #[test]
    fn negative_skewness() {
        let values = vec! [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 3.0, 4.0];

        assert!(skewness(&values) < -1e-3);
    }

    #[test]
    fn played_from_policy() {
        let mut policy = vec! [0.0; 362];
        policy[0] = 1.0;

        assert_eq!(
            format!("{}", Played::from_forward(Color::Black, Point::new(0, 0), 0.5, policy)),
            ";B[aa]V[0.0000]".to_string()
        );
    }

    #[test]
    fn played_from_mcts() {
        let server = Pool::with_capacity(Box::new(FakePredictor::new(1, 0.6)), 1);
        let board = Board::new(0.5);
        let (value, index, tree) =
            predict(
                &server,
                Box::new(StandardDeterministicSearch::default()),
                Box::new(RolloutLimit::new(10)),
                None,
                &board,
                Color::Black
            ).unwrap();

        let point = Point::from_packed_parts(index);
        let played = format!("{}", Played::from_mcts(Color::Black, point, value, &tree));

        assert!(played.contains(";B[ba]"), "{}", played);
        assert!(played.contains("TR[ba]"), "{}", played);
        assert!(played.contains("V["), "{}", played);  // exact value depends on the shape of the rollouts, so we cannot check
        assert!(played.contains("P["), "{}", played);  // exact policy depends on the number of rollouts, so we cannot check
    }

    #[test]
    fn played_pass() {
        assert_eq!(
            format!("{}", Played::pass(Color::Black)),
            ";B[]".to_string()
        );
    }

    #[test]
    fn played_fixed() {
        assert_eq!(
            format!("{}", Played::fixed(Color::Black, Point::new(0, 0))),
            ";B[aa]".to_string()
        );
    }
}
