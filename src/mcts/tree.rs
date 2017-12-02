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
use mcts::spin::Mutex;

use ordered_float::OrderedFloat;
use std::ptr;

lazy_static! {
    /// Mapping from policy index to the `x` coordinate it represents.
    pub static ref X: Box<[u8]> = (0..361).map(|i| (i % 19) as u8).collect::<Vec<u8>>().into_boxed_slice();

    /// Mapping from policy index to the `y` coordinate it represents.
    pub static ref Y: Box<[u8]> = (0..361).map(|i| (i / 19) as u8).collect::<Vec<u8>>().into_boxed_slice();
}

/// A monte carlo search tree.
pub struct Node {
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

    /// The sub-tree that each edge points towards.
    children: [*mut Node; 362]
}

impl Drop for Node {
    fn drop(&mut self) {
        for &child in self.children.iter() {
            if !child.is_null() {
                unsafe { Box::from_raw(child); }
            }
        }
    }
}

impl Node {
    /// Returns an empty search tree with the given starting color and prior
    /// values.
    /// 
    /// # Arguments
    /// 
    /// * `color` - the color of the first players color
    /// * `prior` - the prior values of the nodes
    /// 
    pub fn new(color: Color, prior: Box<[f32]>) -> Node {
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
            children: [ptr::null_mut(); 362]
        }
    }

    /// Returns the best move according to the current search tree. This is
    /// determined as the most visited child.
    pub fn best(&self) -> (f32, usize) {
        let max_i = (0..362).max_by_key(|&i| self.count[i]).unwrap();

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
    fn select(&self) -> usize {
        // compute all UCB1 values in a SIMD friendly manner in the hopes
        // that the compiler will re-write it to make use of modern hardware
        let mut uct = [0.0f32; 368];
        let sqrt_n = (1.0 + self.total_count as f32).sqrt();

        for i in 0..368 {
            const C: f32 = 1.41421356237;
            let exp_bonus = sqrt_n / ((1 + self.count[i]) as f32);

            uct[i] = self.value[i] + self.prior[i] * C * exp_bonus;
        }

        // greedy selection based on the maximum ucb1 value
        (0..362).max_by_key(|&i| OrderedFloat(uct[i])).unwrap()
    }
}

pub type NodeTrace = Vec<(*mut Node, Color, usize)>;

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
pub unsafe fn probe(root: &mut Node, board: &mut Board) -> NodeTrace {
    let mut trace = vec! [];
    let mut current = root;

    loop {
        let next_child = current.select();

        {
            let _guard = current.lock.lock();

            current.total_count += 1;
            current.count[next_child] += 1;
        }

        trace.push((current as *mut Node, current.color, next_child));
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
pub unsafe fn insert(trace: &NodeTrace, color: Color, value: f32, prior: Box<[f32]>) {
    if let Some(&(node, _, index)) = trace.last() {
        let _guard = (*node).lock.lock();

        if (*node).children[index].is_null() {
            (*node).children[index] = Box::into_raw(Box::new(Node::new(color, prior)));
        }
    }

    for &(node, _, index) in trace.iter() {
        let value_ = if color == (*node).color { value } else { -value };

        // incremental update of the average value
        {
            let _guard = (*node).lock.lock();

            (*node).value[index] += (value_ - (*node).value[index]) / ((*node).count[index] as f32);
        }
    }
}
