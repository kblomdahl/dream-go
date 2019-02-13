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

use cpu_time::ProcessTime;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use dg_go::{Board, Color};
use dg_utils::config;
use dg_mcts::predict_service::PredictService;
use dg_mcts::time_control::{TimeStrategy, TimeStrategyResult};
use dg_mcts::tree;
use dg_mcts as mcts;
use dg_nn::Network;

type SearchTree = tree::Node;
type PonderResult = Result<(PredictService, SearchTree, Board, Color), &'static str>;

/// A very simple _time control_ that thinks until a boolean flag is set to
/// `false` or the tree has reached its maximum size.
#[derive(Clone)]
pub struct PonderTimeControl {
    is_running: Arc<AtomicBool>,
    max_tree_size: usize
}

impl TimeStrategy for PonderTimeControl {
    fn try_extend<F: Fn() -> bool>(
        &self,
        root: &SearchTree,
        _predicate: F,
        _factor: f32
    ) -> TimeStrategyResult
    {
        if self.is_running.load(Ordering::SeqCst) {
            let total_visits = root.size();

            if total_visits < self.max_tree_size {
                TimeStrategyResult::NotExpired(self.max_tree_size - total_visits)
            } else {
                TimeStrategyResult::Expired
            }
        } else {
            TimeStrategyResult::Expired
        }
    }
}

/// The worker that performs the pondering in the background. It will keep
/// probing into the given `search_tree`, for the given board state and color
/// until `is_running` is set to false.
/// 
/// # Arguments
/// 
/// * `service` - the neural network service used for inference
/// * `search_tree` - the search tree to probe into
/// * `board` - the board state at the root of the search tree
/// * `to_move` - the color of the player whose turn it is to play
/// * `is_running` - the boolean used to determine when to terminate the search
/// 
fn ponder_worker(
    service: PredictService,
    search_tree: Option<SearchTree>,
    board: Board,
    to_move: Color,
    is_running: Arc<AtomicBool>
) -> (PonderResult, Duration)
{
    let start_time = ProcessTime::now();
    let max_tree_size = (*config::NUM_ROLLOUT).user_defined_or(500_000);
    let result = mcts::predict(
        &service.lock().clone_to_static(),
        None,
        PonderTimeControl { is_running, max_tree_size },
        search_tree,
        &board,
        to_move
    );

    if let Some((_value, _index, next_tree)) = result {
        (Ok((service, next_tree, board, to_move)), start_time.elapsed())
    } else {
        (Err("unrecognized error"), start_time.elapsed())
    }
}

/* -------- PonderService -------- */

/// A service that provides background pondering of the current board state,
/// and allows the user to intercept and replace said pondering state at any
/// point.
pub struct PonderService {
    is_running: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<(PonderResult, Duration)>>,
    last_error: &'static str,
    cpu_time: Duration
}

impl Drop for PonderService {
    fn drop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);

        if let Some(handle) = self.worker.take() {
            let _result = handle.join();
        }
    }
}

impl PonderService {
    /// Returns a service that will ponder, starting from the given board
    /// position.
    /// 
    /// # Arguments
    /// 
    /// * `board` - the initial board.
    ///
    pub fn new(board: Board) -> PonderService {
        let is_running = Arc::new(AtomicBool::new(!*config::NO_PONDER));
        let is_running_worker = is_running.clone();
        let to_move = board.to_move();

        PonderService {
            is_running: is_running,
            worker: Some(thread::spawn(move || {
                if let Some(network) = Network::new() {
                    let service = mcts::predict_service::service(network);

                    ponder_worker(service, None, board, to_move, is_running_worker)
                } else {
                    (Err("unable to load network weights"), Duration::new(0, 0))
                }
            })),
            last_error: "",
            cpu_time: Duration::new(0, 0)
        }
    }

    /// Returns the total amount of time the service has spent pondering in the background, or in
    /// the `service` handler.
    pub fn cpu_time(&self) -> Duration {
        self.cpu_time
    }

    /// Pauses the pondering and gives the caller access to the internal state
    /// through a callback. The pondering will be resumed as soon as the
    /// callback returns.
    /// 
    /// The callback gets the current `search_tree`, `board`, and `color` that
    /// is currently being pondered, and can control what the next ponder target
    /// will be with its return values.
    /// 
    /// # Arguments
    /// 
    /// * `callback` - the callback to execute during the pause
    /// 
    pub fn service<F, T>(&mut self, callback: F) -> Result<T, &'static str>
        where F: FnOnce(&PredictService, SearchTree, (Board, Color)) -> (T, Option<SearchTree>, (Board, Color))
    {
        let handle = match self.worker.take() {
            Some(x) => x,
            None => return Err(self.last_error)
        };

        self.is_running.store(false, Ordering::SeqCst);

        match handle.join().unwrap() {
            (Err(reason), duration) => {
                self.cpu_time += duration;
                self.last_error = reason;

                Err(reason)
            },
            (Ok((service, search_tree, board, to_move)), duration) => {
                let start_time = ProcessTime::now();
                let (result, search_tree, (board, to_move)) = callback(
                    &service,
                    search_tree,
                    (board, to_move)
                );

                // re-spawn the pondering thread now that the callback has been
                // executed.
                let is_running_worker = self.is_running.clone();

                self.cpu_time += start_time.elapsed() + duration;
                self.is_running.store(!*config::NO_PONDER, Ordering::SeqCst);
                self.worker = Some(thread::spawn(move || {
                    ponder_worker(service, search_tree, board, to_move, is_running_worker)
                }));

                Ok(result)
            }
        }
    }

    /// Plays the given move into the current _search tree_. Moving the search
    /// tree forward one turn.
    /// 
    /// # Arguments
    /// 
    /// * `color` - the color of the move
    /// * `coordinate` - `(x, y)` coordinates of the move, or `None` to pass.
    /// 
    pub fn forward(&mut self, color: Color, coordinate: Option<(usize, usize)>) {
        let _result = self.service(move |_service, search_tree, (board, to_move)| {
            let search_tree = if to_move != color {
                // passing moves are not recorded in the GTP protocol, so we
                // will just assume the other player passed once if we are in
                // this situation
                mcts::tree::Node::forward(search_tree, 361)
            } else {
                Some(search_tree)
            };

            // forward the search tree with the given move
            let search_tree = search_tree.and_then(|search_tree| {
                let index = if let Some((x, y)) = coordinate {
                    19 * y + x
                } else {
                    361
                };

                mcts::tree::Node::forward(search_tree, index)
            });

            // forward the board state with the given move
            let other = if let Some((x, y)) = coordinate {
                let mut other = board.clone();

                other.place(color, x, y);
                other
            } else {
                board
            };

            ((), search_tree, (other, color.opposite()))
        });
    }
}
