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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use go::{Board, Color};
use util::config;
use mcts::predict::PredictService;
use mcts::time_control::{TimeStrategy, TimeStrategyResult};
use mcts::tree;
use mcts;
use nn::Network;

type SearchTree = tree::Node;
type PonderResult = Result<(PredictService, SearchTree, Board, Color), &'static str>;

unsafe impl Send for SearchTree {}

/// The maximum size the tree is allowed to grow (in nodes) during pondering.
const MAX_TREE_SIZE: usize = 500_000;

/// A very simple _time control_ that thinks until a boolean flag is set to
/// `false`.
#[derive(Clone)]
pub struct PonderTimeControl {
    is_running: Arc<AtomicBool>
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

            if total_visits < MAX_TREE_SIZE {
                TimeStrategyResult::NotExpired(MAX_TREE_SIZE - total_visits)
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
/// * `next_color` - the color of the player whose turn it is to play
/// * `is_running` - the boolean used to determine when to terminate the search
/// 
fn ponder_worker(
    service: PredictService,
    search_tree: Option<SearchTree>,
    board: Board,
    next_color: Color,
    is_running: Arc<AtomicBool>
) -> PonderResult
{
    let (_, _, next_tree) = mcts::predict::<_>(
        &service.lock(),
        None,
        PonderTimeControl { is_running },
        search_tree,
        &board,
        next_color
    );

    Ok((service, next_tree, board, next_color))
}

/* -------- PonderService -------- */

/// A service that provides background pondering of the current board state,
/// and allows the user to intercept and replace said pondering state at any
/// point.
pub struct PonderService {
    is_running: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<PonderResult>>,
    last_error: &'static str
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
    /// * `next_color` - the color of the player whose turn it is.
    /// 
    pub fn new(board: Board, next_color: Color) -> PonderService {
        let is_running = Arc::new(AtomicBool::new(!*config::NO_PONDER));
        let is_running_worker = is_running.clone();

        PonderService {
            is_running: is_running,
            worker: Some(thread::spawn(move || {
                if let Some(network) = Network::new() {
                    let service = mcts::predict::service(network);

                    ponder_worker(service, None, board, next_color, is_running_worker)
                } else {
                    Err("unable to load network weights")
                }
            })),
            last_error: ""
        }
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
            Err(reason) => {
                self.last_error = reason;

                Err(reason)
            },
            Ok((service, search_tree, board, next_color)) => {
                let (result, search_tree, (board, next_color)) = callback(
                    &service,
                    search_tree,
                    (board, next_color)
                );

                // re-spawn the pondering thread now that the callback has been
                // executed.
                let is_running_worker = self.is_running.clone();

                self.is_running.store(!*config::NO_PONDER, Ordering::SeqCst);
                self.worker = Some(thread::spawn(move || {
                    ponder_worker(service, search_tree, board, next_color, is_running_worker)
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
        let _result = self.service(move |_service, search_tree, (board, next_color)| {
            let search_tree = if next_color != color {
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
