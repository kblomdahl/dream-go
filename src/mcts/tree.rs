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

use go::{Board, Color};
use mcts::param::Param;
use mcts::spin::Mutex;

use ordered_float::OrderedFloat;
use std::ptr;

lazy_static! {
    /// Mapping from policy index to the `x` coordinate it represents.
    pub static ref X: Box<[u8]> = (0..361).map(|i| (i % 19) as u8).collect::<Vec<u8>>().into_boxed_slice();

    /// Mapping from policy index to the `y` coordinate it represents.
    pub static ref Y: Box<[u8]> = (0..361).map(|i| (i / 19) as u8).collect::<Vec<u8>>().into_boxed_slice();
}

pub trait Value {
    unsafe fn update<C: Param, E: Value>(trace: &NodeTrace<E>, color: Color, value: f32);
    fn get<C: Param, E: Value>(node: &Node<E>, index: usize) -> f32;
}

/// An implementation of the _Rapid Action Value Estimation_ heuristic
/// with a minimum MSE schedule as suggested by Sylvain Gelly and
/// David Silver [1].
/// 
/// [1] http://www.cs.utexas.edu/~pstone/Courses/394Rspring13/resources/mcrave.pdf
#[derive(Clone)]
#[allow(dead_code)]
pub struct RAVE;

impl Value for RAVE {
    #[inline]
    unsafe fn update<C: Param, E: Value>(trace: &NodeTrace<E>, color: Color, value: f32) {
        PUCT::update::<C, E>(trace, color, value);

        for (i, &(node, node_color, index)) in trace.iter().enumerate() {
            let value_ = if color == (*node).color { value } else { -value };

            for &(other, other_color, _) in trace.iter().take(i) {
                if node_color == other_color {
                    let _guard = (*other).lock.lock();

                    (*other).amaf_count[index] += 1;
                    (*other).amaf[index] += (value_ - (*other).amaf[index]) / ((*other).amaf_count[index] as f32);
                }
            }
        }
    }

    #[inline]
    fn get<C: Param, E: Value>(node: &Node<E>, index: usize) -> f32 {
        let b_sqr = C::rave_bias() * C::rave_bias();

        // minimum MSE schedule
        if node.count[index] == 0 && node.amaf_count[index] == 0 {
            node.value[index]
        } else {
            let count = node.count[index] as f32;
            let amaf_count = node.amaf_count[index] as f32;
            let beta = amaf_count / (count + amaf_count + 4.0f32*count*amaf_count*b_sqr);

            assert!(0.0 <= beta && beta <= 1.0);

            (1.0f32 - beta) * node.value[index] + beta * node.amaf[index]
        }
    }
}

/// An implementation of the _Polynomial UCT_ as suggested in the AlphaGo Zero
/// paper [1].
/// 
/// [1] https://www.nature.com/articles/nature24270
#[derive(Clone)]
#[allow(dead_code)]
pub struct PUCT;

impl Value for PUCT {
    #[inline]
    unsafe fn update<C: Param, E: Value>(trace: &NodeTrace<E>, color: Color, value: f32) {
        for &(node, _, index) in trace.iter() {
            let value_ = if color == (*node).color { value } else { -value };

            // incremental update of the average value
            let _guard = (*node).lock.lock();

            (*node).value[index] += (value_ - (*node).value[index]) / ((*node).count[index] as f32);
        }
    }

    #[inline]
    fn get<C: Param, E: Value>(node: &Node<E>, index: usize) -> f32 {
        node.value[index]
    }
}

/// A monte carlo search tree.
pub struct Node<E: Value> {
    /// Spinlock used to protect the data in this node during modifications.
    lock: Mutex,

    /// The color of each edge.
    color: Color,

    /// The total number of times any edge has been traversed.
    total_count: i32,

    /// The number of times each edge has been traversed
    count: [i32; 368],

    /// The prior value of each edge as indicated by the policy.
    prior: [f32; 368],

    /// The average value for the sub-tree of each edge.
    value: [f32; 368],

    /// The average value for the all moves as first heuristic of each edge.
    amaf: [f32; 368],

    /// The total number of all moves as first updates each edge has received.
    amaf_count: [i32; 368],

    /// The sub-tree that each edge points towards.
    children: [*mut Node<E>; 362]
}

impl<E: Value> Drop for Node<E> {
    fn drop(&mut self) {
        for &child in self.children.iter() {
            if !child.is_null() {
                unsafe { Box::from_raw(child); }
            }
        }
    }
}

impl<E: Value> Node<E> {
    /// Returns an empty search tree with the given starting color and prior
    /// values.
    /// 
    /// # Arguments
    /// 
    /// * `color` - the color of the first players color
    /// * `prior` - the prior values of the nodes
    /// 
    pub fn new(color: Color, prior: Box<[f32]>) -> Node<E> {
        assert_eq!(prior.len(), 362);

        // copy the prior values into an array size that is dividable
        // by 16 to ensure we can use 256-bit wide SIMD registers.
        let mut prior_padding = [0.0f32; 368];

        for i in 0..362 {
            prior_padding[i] = prior[i];
        }

        Node {
            lock: Mutex::new(),
            color: color,
            total_count: 0,
            count: [0; 368],
            prior: prior_padding,
            value: [0.0f32; 368],
            amaf: [0.0f32; 368],
            amaf_count: [0; 368],
            children: [ptr::null_mut(); 362]
        }
    }

    /// Returns the best move according to the current search tree. This is
    /// determined as the most visited child.
    pub fn best(&self) -> (f32, usize) {
        let max_i = (0..362).max_by_key(|&i| self.count[i]).unwrap();

        if self.amaf_count[max_i] > 0 {
            eprintln!("value = {:.3}, amaf = {:.3}, diff = {:.3}",
                self.value[max_i],
                self.amaf[max_i],
                (self.amaf[max_i] - self.value[max_i]).abs(),
            );
        }

        (self.value[max_i], max_i)
    }

    /// Returns the best move according to the prior value of the root node.
    pub fn prior(&self) -> (f32, usize) {
        let max_i = (0..362).max_by_key(|&i| OrderedFloat(self.prior[i])).unwrap();

        (self.prior[max_i], max_i)
    }

    /// Returns a vector containing the _correct_ normalized probability that each move
    /// should be played given the current search tree.
    pub fn softmax(&self) -> Box<[f32]> {
        let mut s = vec! [0.0f32; 362];
        let mut s_total = 0.0f32;

        for i in 0..362 {
            s[i] = self.count[i] as f32;
            s_total += self.count[i] as f32;
        }

        for i in 0..362 {
            s[i] /= s_total;
        }

        s.into_boxed_slice()
    }

    /// Returns the child with the maximum UCT value.
    fn select<C: Param>(&self) -> usize {
        // compute all UCB1 values in a SIMD friendly manner in the hopes
        // that the compiler will re-write it to make use of modern hardware
        let mut uct = [0.0f32; 368];
        let sqrt_n = (1.0 + self.total_count as f32).sqrt();

        for i in 0..368 {
            let exp_bonus = sqrt_n / ((1 + self.count[i]) as f32);

            uct[i] = E::get::<C, E>(self, i) + self.prior[i] * C::exploration_rate() * exp_bonus;
        }

        // greedy selection based on the maximum ucb1 value
        (0..362).max_by_key(|&i| OrderedFloat(uct[i])).unwrap()
    }
}

pub type NodeTrace<E> = Vec<(*mut Node<E>, Color, usize)>;

/// Probe down the search tree, while updating the given board with the
/// moves the traversed edges represents, and return a list of the
/// edges. Which edges to traverse are determined according to the UCT
/// algorithm.
/// 
/// # Arguments
/// 
/// * `root` - the search tree to probe into
/// * `board` - the board to update with the traversed moves
/// 
pub unsafe fn probe<C, E>(root: &mut Node<E>, board: &mut Board) -> NodeTrace<E>
    where C: Param, E: Value
{
    let mut trace = vec! [];
    let mut current = root;

    loop {
        let next_child = current.select::<C>();

        {
            let _guard = current.lock.lock();

            current.total_count += 1;
            current.count[next_child] += 1;
        }

        trace.push((current as *mut Node<E>, current.color, next_child));
        if next_child != 361 {  // not a passing move
            let (x, y) = (X[next_child] as usize, Y[next_child] as usize);

            debug_assert!(board.is_valid(current.color, x, y));
            board.place(current.color, x, y);
        }

        //
        let child = {
            let _guard = current.lock.lock();

            current.children[next_child]
        };

        if child.is_null() {
            break
        } else {
            current = &mut *child;
        }
    }

    trace
}

/// Insert a new node at the end of the given trace and perform the backup pass
/// updating the average and AMAF values of all nodes in the trace.
/// 
/// # Arguments
/// 
/// * `trace` -
/// * `color` -
/// * `value` -
/// * `prior` -
/// 
pub unsafe fn insert<C, E>(trace: &NodeTrace<E>, color: Color, value: f32, prior: Box<[f32]>)
    where C: Param, E: Value
{
    if let Some(&(node, _, index)) = trace.last() {
        let _guard = (*node).lock.lock();

        if (*node).children[index].is_null() {
            (*node).children[index] = Box::into_raw(Box::new(Node::new(color, prior)));
        }
    }

    E::update::<C, E>(trace, color, value);
}
