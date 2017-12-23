// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

mod dirichlet;
mod global_cache;
pub mod param;
mod spin;
pub mod tree;

use rand::{thread_rng, Rng};
use std::cell::UnsafeCell;
use std::fmt;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use time;

use go::{symmetry, Board, Color};
use mcts::param::*;
use nn::{self, Network, WorkspaceGuard};
use util::b85;
use util::f16::*;

/// Mapping from 1D coordinate to letter used to represent that coordinate in
/// the SGF file format.
const SGF_LETTERS: [char; 26] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
    'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
    'w', 'x', 'y', 'z'
];

pub enum GameResult {
    Resign(String, Board, Color, f32),
    Ended(String, Board)
}

impl fmt::Display for GameResult {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let now = time::now_utc();
        let iso8601 = time::strftime("%Y-%m-%dT%H:%M:%S%z", &now).unwrap();

        match *self {
            GameResult::Resign(ref sgf, _, winner, _) => {
                write!(fmt, "(;GM[1]FF[4]DT[{}]SZ[19]RU[Chinese]KM[7.5]RE[{}+Resign]{})", iso8601, winner, sgf)
            },
            GameResult::Ended(ref sgf, ref board) => {
                let (black, white) = board.get_score();
                let black = black as f32;
                let white = white as f32 + 7.5;
                let winner = {
                    if black > white {
                        format!("B+{:.1}", black - white)
                    } else if white > black {
                        format!("W+{:.1}", white - black)
                    } else {
                        format!("0")
                    }
                };

                write!(fmt, "(;GM[1]FF[4]DT[{}]SZ[19]RU[Chinese]KM[7.5]RE[{}]{})", iso8601, winner, sgf)
            }
        }
    }
}

pub enum PolicyRequest<T: From<f32> + Clone> {
    Ask(Box<[T]>, Sender<(T, Box<[T]>)>),
    Wait(Sender<()>),
    Done
}

/// An abstraction that hides the exact details of how a neural network forward
/// pass is implemented. There are two main implementation `ImmediateForward` and
/// `RemoteForward`, where the former performs the forward pass immediate with no
/// batching and the second forwards it to another worker threads that performs
/// the actual work.
/// 
/// All methods appears to be synchronous but may sleep due to communication with
/// other threads.
trait Forwarder<T: From<f32> + Clone> {
    /// Perform a forward pass of a neural network with the given features
    /// and returns the value and policy.
    /// 
    /// # Arguments
    /// 
    /// * `features` -
    /// 
    fn forward(&mut self, features: Box<[T]>) -> (T, Box<[T]>);
}

/// An implementation of `Forwarder` that performs the forward pass immedietly on
/// a local `nn::Workspace` with batch size `1`.
struct ImmediateForward<'a> {
    workspace: WorkspaceGuard<'a>
}

impl<'a> ImmediateForward<'a> {
    fn new(network: &'a Network) -> ImmediateForward {
        ImmediateForward {
            workspace: network.get_workspace(1)
        }
    }
}

impl<'a, T: From<f32> + Clone> Forwarder<T> for ImmediateForward<'a> {
    fn forward(&mut self, features: Box<[T]>) -> (T, Box<[T]>) {
        let (values, mut policies) = nn::forward::<T>(&mut self.workspace, &vec! [features]);
        let policy = policies.pop().unwrap();

        (values[0].clone(), policy)
    }
}

/// An implementation of `Forwarder` that sends the received features over a
/// channel and relies on the remote endpoint performing the forward
/// pass (presumably with some batching).
struct RemoteForward<T: From<f32> + Clone> {
    remote: Sender<PolicyRequest<T>>
}

impl<T: From<f32> + Clone> Forwarder<T> for RemoteForward<T> {
    fn forward(&mut self, features: Box<[T]>) -> (T, Box<[T]>) {
        let (sender, receiver) = channel();

        self.remote.send(PolicyRequest::Ask(features, sender)).unwrap();
        let (value, policy) = receiver.recv().unwrap();

        (value, policy)
    }
}

/// Performs a forward pass through the neural network for the given board
/// position using a random symmetry to increase entropy.
/// 
/// # Arguments
/// 
/// * `workspace` - the workspace to use during the forward pass
/// * `board` - the board position
/// * `color` - the current player
/// 
fn forward<C, A, T>(agent: &mut A, board: &Board, color: Color) -> (f32, Box<[f32]>)
    where C: Param, A: Forwarder<T>, T: From<f32> + Copy, f32: From<T>
{
    lazy_static! {
        static ref SYMM: Vec<symmetry::Transform> = vec! [
            symmetry::Transform::Identity,
            symmetry::Transform::FlipLR,
            symmetry::Transform::FlipUD,
            symmetry::Transform::Transpose,
            symmetry::Transform::TransposeAnti,
            symmetry::Transform::Rot90,
            symmetry::Transform::Rot180,
            symmetry::Transform::Rot270,
        ];
    }

    global_cache::get_or_insert(board, color, || {
        // pick a random transformation to apply to the features. This is done
        // to increase the entropy of the game slightly and to ensure the engine
        // learns the game is symmetric (which should help generalize)
        let t = *thread_rng().choose(&SYMM).unwrap();
        let features = board.get_features::<T>(color, t);

        // run a forward pass through the network using this transformation
        // and when we are done undo it using the opposite.
        let (value, mut original_policy) = agent.forward(features);

        symmetry::apply(&mut original_policy, t.inverse());

        // copy the policy and replace any invalid moves in the suggested policy
        // with -Inf, while keeping the pass move (361) untouched so that there
        // is always at least one valid move.
        let mut policy = vec! [0.0f32; 362];

        for i in 0..361 {
            let (x, y) = (tree::X[i] as usize, tree::Y[i] as usize);

            if !board.is_valid(color, x, y) {
                policy[i] = ::std::f32::NEG_INFINITY;
            } else {
                policy[i] = f32::from(original_policy[i]);
            }
        }

        // get ride of symmetric moves, this is mostly useful for the opening.
        // Once we are past the first ~7 moves the board is usually sufficiently
        // asymmetric for this to turn into a no-op.
        //
        // we skip the first symmetry because it is the identity symmetry, which
        // is always a symmetry for any board.
        for &t in &SYMM[1..8] {
            if !symmetry::is_symmetric(board, t) {
                continue;
            }

            // figure out which are the useful vertices by eliminating the
            // symmetries from the board.
            let mut visited = [false; 368];

            for i in 0..361 {
                let j = t.apply(i);

                if i != j && !visited[i] {
                    visited[i] = true;
                    visited[j] = true;

                    let src = ::std::cmp::max(i, j);
                    let dst = ::std::cmp::min(i, j);

                    if policy[src].is_finite() {
                        assert!(policy[dst].is_finite());

                        policy[dst] += policy[src];
                        policy[src] = ::std::f32::NEG_INFINITY;
                    }
                }
            }
        }

        // renormalize the policy so that it sums to one after all the pruning that
        // we have performed.
        let policy_sum: f32 = policy.iter().filter(|p| p.is_finite()).sum();

        if policy_sum > 1e-4 {  // do not divide by zero
            let policy_recip = policy_sum.recip();

            for i in 0..362 {
                policy[i] *= policy_recip;
            }
        }

        (f32::from(value), policy.into_boxed_slice())
    })
}

/// The shared variables between the master and each worker thread in the `predict` function.
#[derive(Clone)]
struct ThreadContext<E: tree::Value + Clone, T: From<f32> + Copy> {
    /// The root of the monte carlo tree.
    root: Arc<UnsafeCell<tree::Node<E>>>,

    /// The initial board position at the root the tree.
    starting_point: Board,

    /// The channel to use when communicating features to the cuDNN worker thread.
    sender: Sender<PolicyRequest<T>>,

    /// The number of probes that still needs to be done into the tree.
    remaining: Arc<AtomicIsize>,
}

unsafe impl<E: tree::Value + Clone, T: From<f32> + Copy> Send for ThreadContext<E, T> { }

/// Worker that probes into the given monte carlo search tree until the context
/// is exhausted.
/// 
/// # Arguments
/// 
/// * `context` -
/// 
fn predict_worker<C, E, T>(context: ThreadContext<E, T>)
    where C: Param + Clone + 'static,
          E: tree::Value + Clone + 'static,
          T: From<f32> + Copy + Default + 'static, f32: From<T>
{
    let server = context.sender.clone();
    let mut remote = RemoteForward::<T> { remote: context.sender };

    while context.remaining.fetch_sub(1, Ordering::SeqCst) > 0 {
        loop {
            let mut board = context.starting_point.clone();
            let trace = unsafe { tree::probe::<C, E>(&mut *context.root.get(), &mut board) };

            if let Some(trace) = trace {
                let &(_, color, _) = trace.last().unwrap();
                let next_color = color.opposite();
                let (value, policy) = forward::<C, RemoteForward<T>, T>(&mut remote, &board, next_color);

                unsafe {
                    tree::insert::<C, E>(&trace, next_color, value, policy);
                    break
                }
            } else {
                let (tx, rx) = channel();

                server.send(PolicyRequest::Wait(tx)).unwrap();
                rx.recv().unwrap();
            }
        }
    }

    server.send(PolicyRequest::Done).unwrap();
}

/// Worker that listens for requests to compute the neural network, collect them
/// into batches as much as possible, and then evaluate and send the correct
/// response back to each MCTS worker.
/// 
/// # Arguments
/// 
/// * `network` - the neural network to use for evaluations
/// * `receiver` - the channel to listen for requests on
/// 
fn predict_serve<'a, C, T>(
    network: &Network,
    receiver: Receiver<PolicyRequest<T>>
)
    where C: Param + 'static,
          T: From<f32> + Copy + 'static, f32: From<T>
{
    let mut workspaces = (0..C::batch_size())
        .map(|s| network.get_workspace(s + 1))
        .collect::<Vec<WorkspaceGuard>>();
    let mut remaining_workers = C::thread_count();

    while remaining_workers > 0 {
        // collect a full batch worth of features from the workers, or if the
        // workers gets stuck because they want to probe into a sub-tree that
        // is currently being expanded get as large of a batch as possible.
        let mut features_list = vec! [];
        let mut sender_list = vec! [];
        let mut waiting_list = vec! [];

        while features_list.len() < C::batch_size() {
            match receiver.recv().unwrap() {
                PolicyRequest::Ask(features, sender) => {
                    features_list.push(features);
                    sender_list.push(sender);
                }
                PolicyRequest::Wait(sender) => {
                    if features_list.is_empty() {
                        // if we have no features, then this _should_
                        // be about a request from the last evaluation.
                        sender.send(()).unwrap();
                    } else {
                        waiting_list.push(sender);
                    }
                }
                PolicyRequest::Done => {
                    remaining_workers -= 1;
                },
            }

            if remaining_workers == waiting_list.len() + features_list.len() {
                break;
            }
        }

        debug_assert!(features_list.len() == sender_list.len());

        if !features_list.is_empty() {
            // process the features and the send them back to the worker who
            // sent it using the OneShot channel.
            let workspace = &mut workspaces[features_list.len() - 1];
            let (values, policies) = nn::forward::<T>(workspace, &features_list);

            for ((value, policy), sender) in values.into_iter().zip(policies.into_iter()).zip(sender_list.into_iter()) {
                sender.send((value, policy)).unwrap();
            }

            // awaken any workers waiting for updates before continuing
            for waiting in waiting_list.into_iter() {
                waiting.send(()).unwrap();
            }
        }
    }
}

/// Predicts the _best_ next move according to the given neural network when applied
/// to a monte carlo tree search.
/// 
/// # Arguments
/// 
/// * `network` -
/// * `starting_point` -
/// * `starting_color` -
/// 
pub fn predict_aux<C, E, T>(
    network: &Network,
    starting_tree: Option<tree::Node<E>>,
    starting_point: &Board,
    starting_color: Color
) -> (f32, usize, tree::Node<E>)
    where C: Param + Clone + 'static,
          E: tree::Value + Clone + 'static,
          T: From<f32> + Copy + Default + 'static, f32: From<T>
{
    // if we have a starting tree given, then re-use that tree (after some sanity
    // checks), otherwise we need to query the neural network about what the
    // prior value should be at the root node.
    let mut starting_tree = if let Some(mut starting_tree) = starting_tree {
        assert_eq!(starting_tree.color, starting_color);

        starting_tree
    } else {
        let mut immediate = ImmediateForward::new(network);
        let (_, mut policy) = forward::<C, ImmediateForward, T>(
            &mut immediate,
            starting_point,
            starting_color
        );

        tree::Node::new(starting_color, policy)
    };

    // add some dirichlet noise to the root node of the search tree in order to increase
    // the entropy of the search and avoid overfitting to the prior value
    dirichlet::add::<C>(&mut starting_tree.prior, 0.03);

    // start-up all of the worker threads, and then start listening for requests on the
    // channel we gave each thread.
    let (sender, receiver) = channel();
    let context: ThreadContext<E, T> = ThreadContext {
        root: Arc::new(UnsafeCell::new(starting_tree)),
        starting_point: starting_point.clone(),
        sender: sender,

        remaining: Arc::new(AtomicIsize::new(C::iteration_limit() as isize)),
    };

    let handles = (0..C::thread_count()).map(|_| {
        let context = context.clone();

        thread::spawn(move || { predict_worker::<C, E, T>(context) })
    }).collect::<Vec<JoinHandle<()>>>();

    predict_serve::<C, T>(network, receiver);

    // wait for all threads to terminate to avoid any zombie processes
    for handle in handles.into_iter() { handle.join().unwrap(); }

    assert_eq!(Arc::strong_count(&context.root), 1);

    unsafe {
        let root = UnsafeCell::into_inner(Arc::try_unwrap(context.root).ok().expect(""));
        let (value, index) = root.best(if starting_point.count() < 8 {
            C::temperature()
        } else {
            0.0
        });

        #[cfg(feature = "trace-mcts")]
        eprintln!("{}", tree::to_sgf::<C, E>(&root, starting_point));

        (value, index, root)
    }
}

/// Predicts the _best_ next move according to the given neural network when applied
/// to a monte carlo tree search.
/// 
/// # Arguments
/// 
/// * `network` -
/// * `starting_point` -
/// * `starting_color` -
/// 
pub fn predict<C: Param + Clone + 'static, E: tree::Value + Clone + 'static>(
    network: &Network,
    starting_tree: Option<tree::Node<E>>,
    starting_point: &Board,
    starting_color: Color
) -> (f32, usize, tree::Node<E>)
{
    if network.is_half() {
        predict_aux::<C, E, f16>(network, starting_tree, starting_point, starting_color)
    } else {
        predict_aux::<C, E, f32>(network, starting_tree, starting_point, starting_color)
    }
}

/// Play a game against the engine and return the result of the game.
/// 
/// # Arguments
/// 
/// * `network` - the neural network to use during evaluation
/// 
pub fn self_play(network: &Network) -> GameResult {
    let mut board = Board::new();
    let mut sgf = String::new();
    let mut current = Color::Black;
    let mut pass_count = 0;
    let mut count = 0;

    // limit the maximum number of moves to `2 * 19 * 19` to avoid the
    // engine playing pointless capture sequences at the end of the game
    // that does not change the final result.
    let mut root = None;

    while count < 722 {
        let (value, index, tree) = predict::<Standard, tree::DefaultValue>(
            network,
            root,
            &board,
            current
        );

        debug_assert!(-1.0 <= value && value <= 1.0);
        debug_assert!(index < 362);

        let policy = tree.softmax();
        let (_, prior_index) = tree.prior();

        if value < -0.9 {  // resign the game if the evaluation looks bad
            return GameResult::Resign(sgf, board, current.opposite(), -value);
        } else if index == 361 {  // passing move
            sgf += &format!(";{}[]P[{}]", current, b85::encode(&policy));
            pass_count += 1;

            if pass_count >= 2 {
                return GameResult::Ended(sgf, board)
            }

            root = tree::Node::forward(tree, 361);
        } else {
            let (x, y) = (tree::X[index] as usize, tree::Y[index] as usize);

            sgf += &format!(";{}[{}{}]P[{}]",
                current,
                SGF_LETTERS[x], SGF_LETTERS[y],
                b85::encode(&policy)
            );
            if prior_index != 361 {
                sgf += &format!("TR[{}{}]",
                    SGF_LETTERS[tree::X[prior_index] as usize],
                    SGF_LETTERS[tree::Y[prior_index] as usize]
                );
            };

            pass_count = 0;
            board.place(current, x, y);
            root = tree::Node::forward(tree, index);
        }

        current = current.opposite();
        count += 1;
    }

    GameResult::Ended(sgf, board)
}

/// Play a game against the engine and return the results of the game over
/// the returned channel. This is different from `send_play` because this
/// method does not perform any search and only plays stochastically according
/// to the policy network.
/// 
/// # Arguments
/// 
/// * `network` - the neural network to use during evaluation
/// 
pub fn policy_play_aux<C, T>(
    network: Network
) -> Receiver<GameResult>
    where C: Param + 'static,
          T: From<f32> + Copy + Send + 'static, f32: From<T>
{
    let (sender, receiver) = channel();

    // spawn the worker threads that generate the self-play games
    let (policy_sender, policy_receiver) = channel();

    for _ in 0..(C::thread_count()) {
        let policy_sender = policy_sender.clone();
        let sender = sender.clone();

        thread::spawn(move || {
            let mut remote = RemoteForward::<T> { remote: policy_sender };

            loop {
                let mut board = Board::new();
                let mut sgf = String::new();
                let mut current = Color::Black;
                let mut pass_count = 0;
                let mut count = 0;

                while pass_count < 2 && count < 722 && !board.is_scoreable() {
                    let (_, policy) = forward::<C, RemoteForward<T>, T>(
                        &mut remote,
                        &board,
                        current
                    );

                    // pick a move stochastically according to its prior value, we
                    // do not need to compute the sum because `forward` always
                    // returns a normalized vector of prior values
                    let index = {
                        let threshold = thread_rng().next_f32();
                        let mut so_far = 0.0f32;
                        let mut best = None;

                        for i in 0..362 {
                            if policy[i].is_finite() {
                                so_far += policy[i];

                                if so_far >= threshold {
                                    best = Some(i);
                                    break
                                }
                            }
                        }

                        best  // if nothing, then pass
                    };

                    if let Some(index) = index {
                        if index == 361 {  // pass
                            sgf += &format!(";{}[]", current);
                            pass_count += 1;
                        } else {  // normal move
                            let (x, y) = (tree::X[index] as usize, tree::Y[index] as usize);

                            sgf += &format!(";{}[{}{}]", current, SGF_LETTERS[x], SGF_LETTERS[y]);
                            pass_count = 0;
                            board.place(current, x, y);
                        }
                    } else {  // no valid moves remaining
                        sgf += &format!(";{}[]", current);
                        pass_count += 1;
                    }

                    // continue with the next turn
                    current = current.opposite();
                    count += 1;
                }

                // if the receiver has terminated then quit
                if sender.send(GameResult::Ended(sgf, board)).is_err() {
                    break;
                }
            }

            remote.remote.send(PolicyRequest::Done).unwrap();
        });
    }

    // spawn the server thread that computes the policies for the workers
    thread::spawn(move || { predict_serve::<C, T>(&network, policy_receiver) });

    receiver
}

/// Play a game against the engine and return the results of the game over
/// the returned channel. This is different from `send_play` because this
/// method does not perform any search and only plays stochastically according
/// to the policy network.
/// 
/// # Arguments
/// 
/// * `network` - the neural network to use during evaluation
/// 
pub fn policy_play(network: Network) -> Receiver<GameResult> {
    let is_half = network.is_half();

    if is_half {
        policy_play_aux::<Standard, f16>(network)
    } else {
        policy_play_aux::<Standard, f32>(network)
    }
}
