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

use go::sgf::SgfCoordinate;
use go::{Board, Color};
use mcts::spin::Mutex;
use mcts::argmax::argmax;
use util::{config, max};

use ordered_float::OrderedFloat;
use rand::{thread_rng, Rng};
use std::fmt;
use std::ptr;

lazy_static! {
    /// Mapping from policy index to the `x` coordinate it represents.
    pub static ref X: Box<[u8]> = (0..361).map(|i| (i % 19) as u8).collect::<Vec<u8>>().into_boxed_slice();

    /// Mapping from policy index to the `y` coordinate it represents.
    pub static ref Y: Box<[u8]> = (0..361).map(|i| (i / 19) as u8).collect::<Vec<u8>>().into_boxed_slice();
}

/// An implementation of the _Polynomial UCT_ as suggested in the AlphaGo Zero
/// paper [1].
/// 
/// [1] https://www.nature.com/articles/nature24270
#[derive(Clone)]
pub struct PUCT;

impl PUCT {
    #[inline(always)]
    unsafe fn get_impl(node: &Node, value: &[f32], dst: &mut [f32]) {
        use std::intrinsics::{fadd_fast, fdiv_fast, fmul_fast};

        let sqrt_n = ((1 + node.total_count) as f32).sqrt();
        let uct_exp = *config::UCT_EXP;

        for i in 0..362 {
            let count = *node.count.get_unchecked(i);
            let prior = *node.prior.get_unchecked(i);
            let value = *value.get_unchecked(i);
            let exp_bonus = fdiv_fast(sqrt_n, (1 + count) as f32);

            *dst.get_unchecked_mut(i) = fadd_fast(value, fmul_fast(fmul_fast(prior, uct_exp), exp_bonus));
        }
    }

    #[allow(unused_attributes)]
    #[target_feature(enable = "avx,avx2")]
    unsafe fn get_avx2(node: &Node, value: &[f32], dst: &mut [f32]) {
        // since there is no real _trickery_ in this function except the
        // usage of the AVX2 SIMD instructions for addition, multiplication,
        // etc, we rely on LLVM to do all of the optimizations.
        PUCT::get_impl(node, value, dst);
    }

    /// Update the trace backwards with the given value (and color).
    /// 
    /// # Arguments
    /// 
    /// * `trace` - 
    /// * `color` - 
    /// * `value` - 
    /// 
    #[inline]
    unsafe fn update(trace: &NodeTrace, color: Color, value: f32) {
        for &(node, _, index) in trace.iter() {
            let value_ = if color == (*node).color { value } else { 1.0 - value };

            // incremental update of the average value and remove any additional
            // virtual losses we added to the node
            let _guard = (*node).lock.lock();

            (*node).total_count -= *config::VLOSS_CNT - 1;
            (*node).count[index] -= *config::VLOSS_CNT - 1;
            (*node).total_value[index] += value_;
            (*node).value[index] = (*node).total_value[index] / ((*node).count[index] as f32);
        }
    }

    /// Reference implementation of the PUCT value function.
    /// 
    /// # Arguments
    /// 
    /// * `node` -
    /// * `value` - the winrates to use in the calculations
    /// * `dst` - output array for the UCT value
    /// 
    #[cfg(test)]
    #[inline(always)]
    fn get_ref(node: &Node, value: &[f32], dst: &mut [f32]) {
        unsafe { PUCT::get_impl(node, value, dst) };
    }

    /// Optimized implementation of the PUCT value function.
    /// 
    /// # Arguments
    /// 
    /// * `node` -
    /// * `value` - the winrates to use in the calculations
    /// * `dst` - output array for the UCT value
    /// 
    #[inline(always)]
    fn get(node: &Node, value: &[f32], dst: &mut [f32]) {
        if is_x86_feature_detected!("avx2")  {
            unsafe { PUCT::get_avx2(node, value, dst) };
        } else {
            unsafe { PUCT::get_impl(node, value, dst) };
        }
    }
}

/// Returns the weighted n:th percentile of the given array, and the sum of
/// all smaller elements.
/// 
/// # Arguments
/// 
/// * `array` - 
/// * `n` - 
/// 
fn percentile(array: &[i32], total: i32, n: f64) -> (i32, f64) {
    let mut copy = array.to_vec();
    copy.sort_unstable_by_key(|val| -val);

    // step forward in the array until we have accumulated the requested amount
    let max_value = (total as f64) * (1.0 - n);
    let mut so_far = 0.0;

    for val in copy.into_iter() {
        so_far += val as f64;

        if so_far >= max_value {
            return (val, so_far);
        }
    }

    unreachable!();
}

/// A monte carlo search tree.
pub struct Node {
    /// Spinlock used to protect the data in this node during modifications.
    lock: Mutex,

    /// The color of each edge.
    pub color: Color,

    /// The number of consecutive passes to reach this node.
    pub pass_count: i32,

    /// The total number of times any edge has been traversed.
    pub total_count: i32,

    /// The number of times each edge has been traversed
    pub count: [i32; 368],

    /// The prior value of each edge as indicated by the policy.
    pub prior: [f32; 368],

    /// The total sum of all values for the sub-tree of each edge.
    pub total_value: [f32; 368],

    /// The average value for the sub-tree of each edge.
    pub value: [f32; 368],

    /// Whether some thread is currently busy (or is done) expanding the given
    /// child. This is used to avoid the same child being expanded multiple
    /// times by different threads.
    expanding: [bool; 362],

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
    pub fn new(color: Color, value: f32, prior: Box<[f32]>) -> Node {
        assert_eq!(prior.len(), 362);

        // copy the prior values into an array size that is dividable
        // by 16 to ensure we can use 256-bit wide SIMD registers.
        let mut prior_padding = [::std::f32::NEG_INFINITY; 368];

        for i in 0..362 {
            prior_padding[i] = prior[i];
        }

        Node {
            lock: Mutex::new(),
            color: color,
            pass_count: 0,
            total_count: 0,
            count: [0; 368],
            prior: prior_padding,
            total_value: [0.0; 368],
            value: [value; 368],
            expanding: [false; 362],
            children: [ptr::null_mut(); 362]
        }
    }

    /// Returns the total size of this search tree.
    pub fn size(&self) -> usize {
        self.total_count as usize
    }

    fn as_sgf<S: SgfCoordinate>(&self, fmt: &mut fmt::Formatter, meta: bool) -> fmt::Result {
        // annotate the top-10 moves to make it easier to navigate for the
        // user.
        let mut children = (0..362).collect::<Vec<usize>>();
        children.sort_by_key(|&i| -self.count[i]);

        if meta {
            for i in 0..10 {
                let j = children[i];

                if j != 361 && self.count[j] > 0 {
                    lazy_static! {
                        static ref LABELS: Vec<&'static str> = vec! [
                            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
                        ];
                    }

                    write!(fmt, "LB[{}:{}]",
                        S::to_sgf(X[j] as usize, Y[j] as usize),
                        LABELS[i]
                    )?;
                }
            }

            // mark all valid moves with a triangle (for debugging the symmetry code)
            /*
            for i in 0..361 {
                if self.prior[i].is_finite() {
                    write!(fmt, "TR[{}]",
                        S::to_sgf(X[j] as usize, Y[j] as usize),
                    )?;
                }
            }
            */
        }

        let mut uct = [::std::f32::NEG_INFINITY; 368];
        PUCT::get(self, &self.value, &mut uct);

        for i in children {
            // do not output nodes that has not been visited to reduce the
            // size of the final SGF file.
            if self.count[i] == 0 {
                continue;
            }

            write!(fmt, "(")?;
            write!(fmt, ";{}[{}]",
                if self.color == Color::Black { "B" } else { "W" },
                if i == 361 { "tt".to_string() } else { S::to_sgf(X[i] as usize, Y[i] as usize) },
            )?;
            write!(fmt, "C[prior {:.4} value {:.4} (visits {} / total {}) uct {:.4}]",
                self.prior[i],
                self.value[i],
                self.count[i],
                self.total_count,
                uct[i]
            )?;

            unsafe {
                let child = self.children[i];

                if !child.is_null() {
                    (*child).as_sgf::<S>(fmt, meta)?;
                }
            }

            write!(fmt, ")")?;
        }

        Ok(())
    }

    /// Returns the sub-tree that contains the exploration of the given move index.
    /// 
    /// # Argumnets
    /// 
    /// * `self` - the search tree to pluck the child from
    /// * `index` - the move to pluck the sub-tree for
    /// 
    pub fn forward(mut self, index: usize) -> Option<Node> {
        let child = self.children[index];

        if child.is_null() {
            if index == 361 {
                // we need to record that were was a pass so that we have the correct
                // pass count in the root node.
                let prior = vec! [0.0f32; 362].into_boxed_slice();
                let mut next = Node::new(self.color.opposite(), 0.5, prior);
                next.pass_count = self.pass_count + 1;

                Some(next)
            } else {
                None
            }
        } else {
            Some(unsafe {
                self.children[index] = ptr::null_mut();

                ptr::read(child)
            })
        }
    }

    /// Returns the best move according to the current search tree. This is
    /// determined as the most visited child. If the temperature is non-zero
    /// then this process is stochastic, so that the probability that a move
    /// is picked is proportional to its visit count.
    /// 
    /// # Arguments
    /// 
    /// * `temperature` - How random the process should be, if set to +Inf
    ///   then the values are picked completely at random, and if set to 0
    ///   the selection is greedy.
    /// 
    pub fn best(&self, temperature: f32) -> (f32, usize) {
        if temperature <= 9e-2 { // greedy
            let max_i = (0..362).max_by_key(|&i| self.count[i]).unwrap();

            (self.value[max_i], max_i)
        } else {
            let t = (temperature as f64).recip();
            let c_total = self.count.iter().sum::<i32>();
            let (c_threshold, c_total) = percentile(&self.count, c_total, 0.1);
            let mut s = vec! [::std::f64::NAN; 362];
            let mut s_total = 0.0;

            for i in 0..362 {
                let count = self.count[i];

                if count >= c_threshold {
                    s_total += (count as f64 / c_total).powf(t);
                    s[i] = s_total;
                }
            }

            debug_assert!(s_total.is_finite());

            if s_total < ::std::f64::MIN_POSITIVE {
                (0.5, thread_rng().gen_range(0, 362))
            } else {
                let threshold = s_total * thread_rng().gen::<f64>();
                let max_i = (0..362).filter(|&i| s[i] >= threshold).next().unwrap();

                (self.value[max_i], max_i)
            }
        }
    }

    /// Returns the best move according to the prior value of the root node.
    pub fn prior(&self) -> (f32, usize) {
        let max_i = (0..362).max_by_key(|&i| OrderedFloat(self.prior[i])).unwrap();

        (self.prior[max_i], max_i)
    }

    /// Returns a vector containing the _correct_ normalized probability that each move
    /// should be played given the current search tree.
    pub fn softmax<T: From<f32> + Clone>(&self) -> Vec<T> {
        let mut s = vec! [T::from(0.0f32); 362];
        let mut s_total = 0.0f32;

        for i in 0..362 {
            s_total += self.count[i] as f32;
        }

        for i in 0..362 {
            s[i] = T::from(self.count[i] as f32 / s_total);
        }

        s
    }

    /// Remove the given move as a valid choice in this search tree by setting
    /// its `value` to negative infinity.
    /// 
    /// # Arguments
    /// 
    /// * `index` - the index of the child to disqualify
    /// 
    pub fn disqualify(&mut self, index: usize) {
        self.value[index] = ::std::f32::NEG_INFINITY;
        self.count[index] = 0;
    }

    /// Returns the child with the maximum UCT value, and increase its visit count
    /// by one.
    /// 
    /// # Arguments
    /// 
    /// * `apply_fpu` - whether to use the first-play urgency heuristic 
    /// 
    fn select<'a>(&'a mut self, apply_fpu: bool) -> Option<usize> {
        let mut value = self.value.clone();

        if apply_fpu {
            // for unvisited children, attempt to transform the parent `value`
            // into a reasonable value for that child. This is known as the
            // _First Play Urgency_ heuristic, of the ones that has been tried
            // so far this one turns out to be the best:
            //
            // - square root visit count
            // - constant (this is currently used)
            // - zero
            //
            for i in 0..368 {
                if self.count[i] == 0 {
                    value[i] = max(0.0, value[i] - *config::FPU_REDUCE);
                }
            }
        }

        // compute all UCB1 values for each node before trying to figure out which
        // to pick to make it possible to do it with SIMD.
        let mut uct = [::std::f32::NEG_INFINITY; 368];

        PUCT::get(self, &value, &mut uct);

        // greedy selection based on the maximum ucb1 value, failing if someone else
        // is already expanding the node we want to expand.
        let _guard = self.lock.lock();
        let max_i = argmax(&uct).and_then(|i| {
            if self.expanding[i] && self.children[i].is_null() {
                None  // someone else is already expanding this node
            } else {
                Some(i)
            }
        });

        if let Some(max_i) = max_i {
            self.total_count += *config::VLOSS_CNT;
            self.count[max_i] += *config::VLOSS_CNT;
            self.value[max_i] = self.total_value[max_i] / (self.count[max_i] as f32);
            self.expanding[max_i] = true;
        }

        max_i
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
pub unsafe fn probe(root: &mut Node, board: &mut Board) -> Option<NodeTrace> {
    let mut trace = vec! [];
    let mut current = root;

    loop {
        if let Some(next_child) = current.select(!trace.is_empty()) {
            trace.push((current as *mut Node, current.color, next_child));

            if next_child != 361 {  // not a passing move
                let (x, y) = (X[next_child] as usize, Y[next_child] as usize);

                debug_assert!(board.is_valid(current.color, x, y));
                board.place(current.color, x, y);
            } else if current.pass_count >= 1 {
                break;  // at least two consecutive passes
            }

            //
            let child = current.children[next_child];

            if child.is_null() {
                break
            } else {
                current = &mut *child;
            }
        } else {
            // undo the entire trace, since we added virtual losses (optimistically)
            // on the way down.
            for (node, _, next_child) in trace.into_iter() {
                let _guard = (*node).lock.lock();

                (*node).total_count -= *config::VLOSS_CNT;
                (*node).count[next_child] -= *config::VLOSS_CNT;
            }

            return None;
        }
    }

    Some(trace)
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
        let mut next = Box::new(Node::new(color, value, prior));
        if index == 361 {
            next.pass_count = (*node).pass_count + 1;
        }

        if (*node).children[index].is_null() {
            (*node).children[index] = Box::into_raw(next);
        } else {
            debug_assert!(index == 361);

            // since we stop probing into a tree once two consecutive passes has
            // occured we can double-expand those nodes. This is too prevent that
            // from causing memory leaks.
        }
    }

    PUCT::update(trace, color, value);
}

/// Type alias for `Node` that acts as a wrapper for calling `as_sgf` from
/// within a `write!` macro.
pub struct ToSgf<'a, S: SgfCoordinate> {
    _coordinate_format: ::std::marker::PhantomData<S>,
    starting_point: Board,
    root: &'a Node,
    meta: bool
}

impl<'a, S: SgfCoordinate> fmt::Display for ToSgf<'a, S> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        if self.meta {
            // add the standard SGF prefix
            write!(fmt, "(;GM[1]FF[4]SZ[19]RU[Chinese]KM[{:.1}]PL[{}]",
                self.starting_point.komi(),
                if self.root.color == Color::Black { "B" } else { "W" }
            )?;

            // write the starting point to the SGF file as pre-set variables
            for y in 0..19 {
                for x in 0..19 {
                    match self.starting_point.at(x, y) {
                        None => Ok(()),
                        Some(Color::Black) => write!(fmt, "AB[{}]", S::to_sgf(x, y)),
                        Some(Color::White) => write!(fmt, "AW[{}]", S::to_sgf(x, y))
                    }?
                }
            }

            // write the actual search tree
            self.root.as_sgf::<S>(fmt, self.meta)?;

            // add the standard SGF suffix
            write!(fmt, ")")
        } else {
            // write the actual search tree
            self.root.as_sgf::<S>(fmt, self.meta)
        }
    }
}

/// Returns a marker that contains all the examined positions of the given
/// search tree and can be displayed as an SGF file.
/// 
/// # Arguments
/// 
/// * `root` -
/// * `starting_point` -
/// * `meta` - whether to include the SGF meta data (rules, etc.)
/// 
pub fn to_sgf<'a, S>(root: &'a Node, starting_point: &Board, meta: bool) -> ToSgf<'a, S>
    where S: SgfCoordinate
{
    ToSgf {
        _coordinate_format: ::std::marker::PhantomData::default(),
        starting_point: starting_point.clone(),
        root: &root,
        meta: meta
    }
}

/// Type alias for pretty-printing an index based vertex.
struct PrettyVertex {
    inner: usize
}

impl fmt::Display for PrettyVertex {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        if self.inner == 361 {
            fmt.pad("pass")
        } else {
            const LETTERS: [char; 19] = [
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k',
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'
            ];

            fmt.pad(&format!("{}{}",
                LETTERS[X[self.inner] as usize],
                Y[self.inner] + 1
            ))
        }
    }
}

/// Iterator that traverse the most likely path down a search tree
pub struct GreedyPath<'a> {
    current: &'a Node,
}

impl<'a> Iterator for GreedyPath<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let max_i = (0..362).max_by_key(|&i| self.current.count[i]).unwrap();

        if self.current.count[max_i] == 0 {
            None
        } else {
            unsafe {
                self.current = &*self.current.children[max_i];
            }

            Some(max_i)
        }
    }
}

/// Type alias for `Node` that acts as a wrapper for calling `as_sgf` from
/// within a `write!` macro.
pub struct ToPretty<'a> {
    root: &'a Node,
}

impl<'a> fmt::Display for ToPretty<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut children = (0..362).collect::<Vec<usize>>();
        children.sort_by_key(|&i| -self.root.count[i]);

        // print a summary containing the total tree size
        let total_value: f32 = (0..362)
            .map(|i| (self.root.count[i] as f32) * self.root.value[i])
            .sum();
        let norm_value = total_value / (self.root.total_count as f32);
        let likely_path: String = GreedyPath { current: self.root }
                .map(|i| PrettyVertex { inner: i })
                .map(|v| format!("{}", v))
                .collect::<Vec<String>>().join(" ");

        write!(fmt, "Nodes: {}, Win: {:.1}%, PV: {}\n",
            self.root.total_count,
            100.0 * norm_value,
            likely_path
        )?;

        // print a summary of each move that we considered
        for i in children {
            if self.root.count[i] <= 1  {
                continue;
            }

            let pretty_vertex = PrettyVertex { inner: i };
            let child = unsafe { &*self.root.children[i] };
            let likely_path: String = GreedyPath { current: child }
                    .map(|i| PrettyVertex { inner: i })
                    .map(|v| format!("{}", v))
                    .collect::<Vec<String>>().join(" ");

            write!(fmt, "{: >5} -> {:7} (W: {:5.2}%) (N: {:5.2}%) PV: {} {}\n",
                pretty_vertex,
                child.total_count,
                100.0 * self.root.value[i],
                100.0 * self.root.prior[i],
                pretty_vertex,
                likely_path
            )?;
        }

        Ok(())
    }
}

/// Returns a marker that contains all the examined positions of the given
/// search tree and can be pretty-printed to something easily examined by
/// a human.
/// 
/// # Arguments
/// 
/// * `root` -
/// * `starting_point` -
/// 
pub fn to_pretty<'a>(root: &'a Node) -> ToPretty<'a> {
    ToPretty { root: root }
}

#[cfg(test)]
mod tests {
    use test::{self, Bencher};
    use rand::rngs::SmallRng;
    use rand::{Rng, FromEntropy};
    use go::*;
    use mcts::tree::*;

    fn get_prior_distribution(rng: &mut SmallRng, board: Board, color: Color) -> Box<[f32]> {
        let mut prior: Vec<f32> = (0..362).map(|_| rng.gen::<f32>()).collect();
        let sum_recip: f32 = prior.iter().map(|&f| f).sum();

        for i in 0..362 {
            if i == 361 || board.is_valid(color, X[i] as usize, Y[i] as usize) {
                prior[i] *= sum_recip;
            } else {
                prior[i] = ::std::f32::NEG_INFINITY;
            }
        }

        prior.into_boxed_slice()
    }

    unsafe fn bench_test(b: &mut Bencher) {
        let mut rng = SmallRng::from_entropy();
        let mut root = Node::new(Color::Black, 0.5, get_prior_distribution(&mut rng, Board::new(DEFAULT_KOMI), Color::Black));

        for t in 0..800 {
            let mut board = Board::new(DEFAULT_KOMI);
            let mut dst_ref = [0.0f32; 368];
            let mut dst_asm = [0.0f32; 368];

            // check so that the reference and asm implementation gives back the same
            // value
            PUCT::get(&root, &root.value, &mut dst_asm);
            PUCT::get_ref(&root, &root.value, &mut dst_ref);

            for i in 0..362 {
                // because of numeric instabilities and approximations the answers may
                // be slightly different
                const EPS: f32 = 1e-4;

                assert!(
                    (dst_asm[i] - dst_ref[i]).abs() < EPS,
                    "epoch {}: dst[{}] <=> asm {} != ref {}",
                    t, i, dst_asm[i], dst_ref[i]
                );
            }

            // expand the tree by one probe so that the root values change
            let trace = probe(&mut root, &mut board).unwrap();
            let &(_, color, _) = trace.last().unwrap();
            let next_color = color.opposite();
            let (value, policy) = (rng.gen::<f32>(), get_prior_distribution(&mut rng, board, next_color));

            insert(&trace, next_color, value, policy);
        }

        // benchmark the value function only
        let root = test::black_box(root);

        b.iter(|| {
            let mut dst = test::black_box([0.0f32; 368]);

            PUCT::get(&root, &root.value, &mut dst);
            dst
        });

    }

    #[bench]
    fn puct(b: &mut Bencher) {
        unsafe { bench_test(b); }
    }
}
