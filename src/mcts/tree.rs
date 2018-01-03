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
use rand::{thread_rng, Rng};
use std::fmt;
use std::ptr;

/// Mapping from 1D coordinate to letter used to represent that coordinate in
/// the SGF file format.
#[cfg(feature = "trace-mcts")]
const SGF_LETTERS: [char; 26] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
    'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
    'w', 'x', 'y', 'z'
];

lazy_static! {
    /// Mapping from policy index to the `x` coordinate it represents.
    pub static ref X: Box<[u8]> = (0..361).map(|i| (i % 19) as u8).collect::<Vec<u8>>().into_boxed_slice();

    /// Mapping from policy index to the `y` coordinate it represents.
    pub static ref Y: Box<[u8]> = (0..361).map(|i| (i / 19) as u8).collect::<Vec<u8>>().into_boxed_slice();
}

pub trait Value {
    unsafe fn update<C: Param, E: Value>(trace: &NodeTrace<E>, color: Color, value: f32);
    fn get<C: Param, E: Value>(node: &Node<E>, dst: &mut [f32]);
    fn get_ref<C: Param, E: Value>(node: &Node<E>, dst: &mut [f32]);
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
            let value_ = if color == (*node).color { value } else { 1.0 - value };

            for &(other, other_color, _) in trace.iter().take(i) {
                if node_color == other_color {
                    let _guard = (*other).lock.lock();

                    (*other).amaf_count[index] += 1;
                    (*other).amaf[index] += (value_ - (*other).amaf[index]) / ((*other).amaf_count[index] as f32);
                }
            }
        }
    }

    /// Reference implementation of the RAVE value function.
    /// 
    /// # Arguments
    /// 
    /// * `node` -
    /// * `dst` - output array for the UCT value
    /// 
    #[inline]
    fn get_ref<C: Param, E: Value>(node: &Node<E>, dst: &mut [f32]) {
        let sqrt_n = ((1 + node.total_count) as f32).sqrt();
        let b_sqr = C::rave_bias() * C::rave_bias();

        for i in 0..362 {
            let exp_bonus = sqrt_n * ((1 + node.count[i]) as f32).recip();
            let value = if node.count[i] == 0 && node.amaf_count[i] == 0 {
                node.value[i]
            } else {
                // minimum MSE schedule
                let count = node.count[i] as f32;
                let amaf_count = node.amaf_count[i] as f32;
                let beta = amaf_count / (count + amaf_count + 4.0f32*count*amaf_count*b_sqr);

                debug_assert!(0.0 <= beta && beta <= 1.0);

                (1.0f32 - beta) * node.value[i] + beta * node.amaf[i]
            };

            dst[i] = value + node.prior[i] * C::exploration_rate() * exp_bonus
        }
    }

    #[inline]
    fn get<C: Param, E: Value>(node: &Node<E>, dst: &mut [f32]) {
        if cfg!(target_arch = "x86_64") {
            let sqrt_n = ((1 + node.total_count) as f32).sqrt();
            let b_sqr = C::rave_bias() * C::rave_bias();

            #[cfg(target_arch = "x86_64")]
            unsafe {
                const ONE: f32 = 1.0f32;

                asm!(r#"
                    vxorps ymm11, ymm11, ymm11        # ymm11 = 0
                    vbroadcastss ymm12, [rax]         # ymm12 = sqrt (total_count + 1)
                    vbroadcastss ymm13, [rbx]         # ymm13 = 4 * b_sqr
                    vbroadcastss ymm14, [rcx]         # ymm14 = exploration_rate
                    vbroadcastss ymm15, [rdx]         # ymm15 = 1.0
                    mov rcx, 46                       # loop counter
                    xor rax, rax                      # rax = 0

                    1:
                    vmovups ymm0, [ r8+4*rax]         # ymm0 = count[rax]
                    vmovups ymm1, [ r9+4*rax]         # ymm1 = value[rax]
                    vmovups ymm2, [r10+4*rax]         # ymm2 = amaf_count[rax]
                    vmovups ymm3, [r11+4*rax]         # ymm3 = amaf[rax]
                    vmovups ymm4, [r12+4*rax]         # ymm4 = prior[rax]

                    vcvtdq2ps ymm5, ymm0              # ymm5  = ymm0 as f32
                    vaddps ymm6, ymm5, ymm15          # ymm6 += ymm5 + 1
                    vdivps ymm6, ymm12, ymm6          # ymm6  = sqrt_n / ymm6  (=exp_bonus)

                    vcvtdq2ps ymm7, ymm2              # ymm7  = ymm2 as f32
                    vmulps ymm8, ymm13, ymm5          # ymm8  = (4 * b_sqr) * count
                    vmulps ymm8, ymm8, ymm7           # ymm8 *= amaf_count
                    vaddps ymm8, ymm8, ymm7           # ymm8 += amaf_count
                    vaddps ymm8, ymm8, ymm5           # ymm8 += count
                    vrcpps ymm8, ymm8                 # ymm8  = 1 / ymm8
                    vmulps ymm8, ymm7, ymm8           # ymm8 *= amaf_count  (=beta)
                    vsubps ymm9, ymm15, ymm8          # ymm9  = 1.0 - ymm8
                    vmulps ymm9, ymm9, ymm1           # ymm9 *= value
                    vmulps ymm3, ymm8, ymm3           # ymm3 *= ymm8
                    vaddps ymm3, ymm3, ymm9           # ymm3 += ymm9

                    vpcmpeqd ymm7, ymm0, ymm11        # ymm7  = (ymm0 == 0)
                    vpcmpeqd ymm8, ymm2, ymm11        # ymm8  = (ymm2 == 0)
                    vandps ymm8, ymm8, ymm7           # ymm8  = ymm7 && ymm8
                    vblendvps ymm1, ymm3, ymm1, ymm8  # ymm1  = if ymm8 { ymm1 } else { ymm3 }

                    vmulps ymm4, ymm4, ymm14          # ymm4 *= ymm14
                    vmulps ymm4, ymm4, ymm6           # ymm4 *= ymm6
                    vaddps ymm4, ymm4, ymm1           # ymm4 += ymm1

                    vmovups [r13+4*rax], ymm4         # dst[rax] = ymm4

                    add rax, 8                        # rax += 8
                    dec ecx                           # rcx -= 1
                    jnz 1b                            # repeat until rcx = 0
                    "#
                    : // no register outputs, but clobber memory
                    : "{r8}"(node.count.as_ptr()),
                      "{r9}"(node.value.as_ptr()),
                      "{r10}"(node.amaf_count.as_ptr()),
                      "{r11}"(node.amaf.as_ptr()),
                      "{r12}"(node.prior.as_ptr()),
                      "{r13}"(dst.as_ptr()),
                      "{rax}"(&sqrt_n),
                      "{rbx}"(&(4.0f32 * b_sqr)),
                      "{rcx}"(&C::exploration_rate()),
                      "{rdx}"(&ONE)
                    : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
                      "ymm7", "ymm8", "ymm9", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"
                    : "intel", "volatile"
                );
            }
        } else {
            RAVE::get_ref::<C, E>(node, dst);
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
            let value_ = if color == (*node).color { value } else { 1.0 - value };

            // incremental update of the average value
            let _guard = (*node).lock.lock();

            (*node).value[index] += (value_ - (*node).value[index]) / ((*node).count[index] as f32);
        }
    }

    /// Reference implementation of the RAVE value function.
    /// 
    /// # Arguments
    /// 
    /// * `node` -
    /// * `dst` - output array for the UCT value
    /// 
    #[inline]
    fn get_ref<C: Param, E: Value>(node: &Node<E>, dst: &mut [f32]) {
        let sqrt_n = ((1 + node.total_count) as f32).sqrt();

        for i in 0..362 {
            let exp_bonus = sqrt_n * ((1 + node.count[i]) as f32).recip();

            dst[i] = node.value[i] + node.prior[i] * C::exploration_rate() * exp_bonus
        }
    }

    #[inline]
    fn get<C: Param, E: Value>(node: &Node<E>, dst: &mut [f32]) {
        if cfg!(target_arch = "x86_64") {
            let sqrt_n = ((1 + node.total_count) as f32).sqrt();

            #[cfg(target_arch = "x86_64")]
            unsafe {
                const ONE: i32 = 1;

                asm!(r#"
                    vbroadcastss ymm3, [r12]  # ymm3 = exploration_rate
                    vbroadcastss ymm4, [r13]  # ymm4 = sqrt (total_count + 1)
                    vbroadcastss ymm5, [r14]  # ymm5 = 1
                    mov rcx, 46               # loop counter

                    1:
                    vmovups ymm0, [ r8]       # ymm0 = count[i]
                    vmovups ymm1, [ r9]       # ymm1 = value[i]
                    vmovups ymm2, [r10]       # ymm2 = prior[i]

                    vpaddd ymm0, ymm0, ymm5   # count[i] += 1
                    vcvtdq2ps ymm0, ymm0      # count[i] = count[i] as f32
                    vmulps ymm2, ymm2, ymm3   # prior[i] *= exploration_rate
                    vdivps ymm0, ymm4, ymm0   # count[i] = sqrt (total_count + 1) / count[i]
                    vmulps ymm0, ymm0, ymm2   # count[i] *= prior[i]
                    vaddps ymm0, ymm0, ymm1   # count[i] += value[i]
                    vmovups [r11], ymm0       # dst[i] = count[i]

                    add  r8, 32               # i += 32
                    add  r9, 32               # ...
                    add r10, 32               # ...
                    add r11, 32               # ...

                    dec ecx                   # rcx -= 1
                    jnz 1b                    # repeat until rcx = 0
                    "#
                    : // no register outputs, but clobber memory
                    : "{r8}"(node.count.as_ptr()),
                      "{r9}"(node.value.as_ptr()),
                      "{r10}"(node.prior.as_ptr()),
                      "{r11}"(dst.as_ptr()),
                      "{r12}"(&C::exploration_rate()),
                      "{r13}"(&sqrt_n),
                      "{r14}"(&ONE)
                    : "memory", "rcx", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5"
                    : "intel", "volatile"
                );
            }
        } else {
            PUCT::get_ref::<C, E>(node, dst);
        }
    }
}

pub type DefaultValue = PUCT;

/// Returns the index of the maximum value in the given array. If multiple
/// indices share the same value, then which is returned is undefined.
/// 
/// # Arguments
/// 
/// * `array` -
/// 
#[inline(never)]
fn argmax(array: &[f32]) -> Option<usize> {
    // This function is `inline(never)` to avoid LLVM from performing dead
    // code elimination on the inline assembly.

    if cfg!(target_arch = "x86_64") {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            const NEG_INFINITY: [f32; 8] = [
                ::std::f32::NEG_INFINITY, ::std::f32::NEG_INFINITY, ::std::f32::NEG_INFINITY, ::std::f32::NEG_INFINITY,
                ::std::f32::NEG_INFINITY, ::std::f32::NEG_INFINITY, ::std::f32::NEG_INFINITY, ::std::f32::NEG_INFINITY
            ];
            let index: usize;

            asm!(r#"
                vmovups ymm3, [r9]                # ymm3 = -inf
                xor rax, rax                      # rax = 0
                xor rbx, rbx                      # rbx = 0
                xor r10, r10                      # r10 = 0
                mov rcx, 46                       # loop counter

                1:
                vmovups ymm0, [r8+4*r10]          # ymm0 = array[r10]

                # this is a tree reduction of the horizontal maximum of `ymm0`
                # by shuffling the elements around and taking the maximum
                # again. For example:
                #
                # a b c d | e f g h  ymm0
                # b a d c | f e h g  ymm1 = shuffle(ymm0, [1, 0, 3, 2])
                # -----------------  ymm0 = max(ymm0, ymm1)
                # a a c c | e e g g  ymm0
                # c c a a | g g e e  ymm1 = shuffle(ymm0, [2, 3, 0, 1])
                # -----------------  ymm0 = max(ymm0, ymm1)
                # a a a a | e e e e  ymm0
                # e e e e | a a a a  ymm1 = shuffle_hilo(ymm0)
                # -----------------  ymm0 = max(ymm0, ymm1)
                # a a a a | a a a a  ymm0
                #

                vpermilps ymm1, ymm0, 0xb1        # ymm1 = shuffle(ymm0, [1, 0, 3, 2])
                vmaxps ymm2, ymm0, ymm1           # ymm2 = max(ymm1, ymm0)
                vpermilps ymm1, ymm2, 0x4e        # ymm1 = shuffle(ymm2, [2, 3, 0, 1])
                vmaxps ymm2, ymm2, ymm1           # ymm2 = max(ymm2, ymm1)
                vperm2f128 ymm1, ymm2, ymm2, 0x01 # ymm1 = shuffle_hilo(ymm2)
                vmaxps ymm2, ymm2, ymm1           # ymm2 = max(ymm2, ymm1)

                vmaxps ymm3, ymm3, ymm2           # ymm3 = max(ymm3, ymm2)
                vcmpeqps ymm4, ymm3, ymm0         # ymm4 = (ymm0 == ymm3)
                vmovmskps eax, ymm4               # eax  = compressed ymm4

                tzcnt eax, eax                    # eax  = leading zero in eax
                jc 2f                             # if eax == 0 skip
                lea rbx, [r10+rax]

                2:
                add r10, 8                         # i += 8

                dec ecx                           # rcx -= 1
                jnz 1b                            # repeat until rcx = 0
                "#
                : "={rbx}"(index)
                : "{r8}"(array.as_ptr()), "{r9}"(NEG_INFINITY.as_ptr())
                : "rax", "rbx", "rcx", "r10", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4"
                : "intel", "volatile"
            );

            Some(index)
        }
    } else {
        (0..362).filter(|&i| array[i].is_finite())
                .max_by_key(|&i| OrderedFloat(array[i]))
    }
}

/// A monte carlo search tree.
pub struct Node<E: Value> {
    /// Spinlock used to protect the data in this node during modifications.
    lock: Mutex,

    /// The color of each edge.
    pub color: Color,

    /// The number of consecutive passes to reach this node.
    pub pass_count: i32,

    /// The total number of times any edge has been traversed.
    total_count: i32,

    /// The number of times each edge has been traversed
    count: [i32; 368],

    /// The prior value of each edge as indicated by the policy.
    pub prior: [f32; 368],

    /// The average value for the sub-tree of each edge.
    value: [f32; 368],

    /// The average value for the all moves as first heuristic of each edge.
    amaf: [f32; 368],

    /// The total number of all moves as first updates each edge has received.
    amaf_count: [i32; 368],

    /// Whether some thread is currently busy (or is done) expanding the given
    /// child. This is used to avoid the same child being expanded multiple
    /// times by different threads.
    expanding: [bool; 362],

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
            value: [0.0f32; 368],
            amaf: [0.0f32; 368],
            amaf_count: [0; 368],
            expanding: [false; 362],
            children: [ptr::null_mut(); 362]
        }
    }

    #[cfg(feature = "trace-mcts")]
    fn as_sgf<C: Param>(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        // annotate the top-10 moves to make it easier to navigate for the
        // user.
        let mut children = (0..362).collect::<Vec<usize>>();
        children.sort_by_key(|&i| -self.count[i]);

        for i in 0..10 {
            let j = children[i];

            if j != 361 && self.count[j] > 0 {
                lazy_static! {
                    static ref LABELS: Vec<&'static str> = vec! [
                        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
                    ];
                }

                write!(fmt, "LB[{}{}:{}]",
                    SGF_LETTERS[X[j] as usize],
                    SGF_LETTERS[Y[j] as usize],
                    LABELS[i]
                )?;
            }
        }

        // mark all valid moves with a triangle (for debugging the symmetry code)
        /*
        for i in 0..361 {
            if self.prior[i].is_finite() {
                write!(fmt, "TR[{}{}]",
                    SGF_LETTERS[X[i] as usize],
                    SGF_LETTERS[Y[i] as usize],
                )?;
            }
        }
        */

        let mut uct = [::std::f32::NEG_INFINITY; 368];
        E::get::<C, E>(self, &mut uct);

        for i in children {
            // do not output nodes that has not been visited to reduce the
            // size of the final SGF file.
            if self.count[i] == 0 {
                continue;
            }

            write!(fmt, "(")?;
            write!(fmt, ";{}[{}{}]",
                if self.color == Color::Black { "B" } else { "W" },
                if i == 361 { 't' } else { SGF_LETTERS[X[i] as usize] },
                if i == 361 { 't' } else { SGF_LETTERS[Y[i] as usize] }
            )?;
            write!(fmt, "C[prior {:.4} value {:.4} (visits {} / total {}) amaf {:.4} (visits {}) uct {:.4}]",
                self.prior[i],
                self.value[i],
                self.count[i],
                self.total_count,
                self.amaf[i],
                self.amaf_count[i],
                uct[i]
            )?;

            unsafe {
                let child = self.children[i];

                if !child.is_null() {
                    (*child).as_sgf::<C>(fmt)?;
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
    pub fn forward(mut self, index: usize) -> Option<Node<E>> {
        let child = self.children[index];

        if child.is_null() {
            if index == 361 {
                // we need to record that were was a pass so that we have the correct
                // pass count in the root node.
                let prior = vec! [0.0f32; 362].into_boxed_slice();
                let mut next = Node::new(self.color.opposite(), prior);
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
            let mut s = vec! [0.0; 362];
            let mut s_total = 0.0;

            for i in 0..362 {
                let count = (self.count[i] as f64).powf(t);

                s_total += count;
                s[i] = s_total;
            }

            debug_assert!(s_total.is_finite());

            let threshold = s_total * thread_rng().next_f64();
            let max_i = (0..362).filter(|&i| s[i] >= threshold).next().unwrap();

            (self.value[max_i], max_i)
        }
    }

    /// Returns the best move according to the prior value of the root node.
    pub fn prior(&self) -> (f32, usize) {
        let max_i = (0..362).max_by_key(|&i| OrderedFloat(self.prior[i])).unwrap();

        (self.prior[max_i], max_i)
    }

    /// Returns a vector containing the _correct_ normalized probability that each move
    /// should be played given the current search tree.
    pub fn softmax<T: From<f32> + Clone>(&self) -> Box<[T]> {
        let mut s = vec! [T::from(0.0f32); 362];
        let mut s_total = 0.0f32;

        for i in 0..362 {
            s_total += self.count[i] as f32;
        }

        for i in 0..362 {
            s[i] = T::from(self.count[i] as f32 / s_total);
        }

        s.into_boxed_slice()
    }

    /// Returns the child with the maximum UCT value, and increase its visit count
    /// by one.
    fn select<'a, C: Param>(&'a mut self) -> Option<usize> {
        // compute all UCB1 values for each node before trying to figure out which
        // to pick to make it possible to do it with SIMD.
        let mut uct = [::std::f32::NEG_INFINITY; 368];

        E::get::<C, E>(self, &mut uct);

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
            self.total_count += 1;
            self.count[max_i] += 1;
            self.expanding[max_i] = true;
        }

        max_i
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
pub unsafe fn probe<C, E>(root: &mut Node<E>, board: &mut Board) -> Option<NodeTrace<E>>
    where C: Param, E: Value
{
    let mut trace = vec! [];
    let mut current = root;

    loop {
        if let Some(next_child) = current.select::<C>() {
            trace.push((current as *mut Node<E>, current.color, next_child));

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

                (*node).total_count -= 1;
                (*node).count[next_child] -= 1;
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
pub unsafe fn insert<C, E>(trace: &NodeTrace<E>, color: Color, value: f32, prior: Box<[f32]>)
    where C: Param, E: Value
{
    if let Some(&(node, _, index)) = trace.last() {
        let mut next = Box::new(Node::new(color, prior));
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

    E::update::<C, E>(trace, color, value);
}

/// Type alias for `Node<E>` that acts as a wrapper for calling `as_sgf` from
/// within a `write!` macro.
#[cfg(feature = "trace-mcts")]
pub struct ToSgf<'a, C: Param + 'a, E: Value + 'a> {
    value_type: ::std::marker::PhantomData<C>,
    starting_point: Board,
    root: &'a Node<E>,
}

#[cfg(feature = "trace-mcts")]
impl<'a, C: Param + 'a, E: Value + 'a> fmt::Display for ToSgf<'a, C, E> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        // add the standard SGF prefix
        write!(fmt, "(;GM[1]FF[4]SZ[19]RU[Chinese]KM[7.5]PL[{}]",
            if self.root.color == Color::Black { "B" } else { "W" }
        )?;

        // write the starting point to the SGF file as pre-set variables
        for y in 0..19 {
            for x in 0..19 {
                match self.starting_point.at(x, y) {
                    None => Ok(()),
                    Some(Color::Black) => write!(fmt, "AB[{}{}]", SGF_LETTERS[x], SGF_LETTERS[y]),
                    Some(Color::White) => write!(fmt, "AW[{}{}]", SGF_LETTERS[x], SGF_LETTERS[y])
                }?
            }
        }

        // write the actual search tree
        self.root.as_sgf::<C>(fmt)?;

        // add the standard SGF suffix
        write!(fmt, ")")
    }
}

/// Returns a marker that contains all the examined positions of the given
/// search tree and can be displayed as an SGF file.
/// 
/// # Arguments
/// 
/// * `root` -
/// * `starting_point` -
/// 
#[cfg(feature = "trace-mcts")]
pub fn to_sgf<'a, C, E>(root: &'a Node<E>, starting_point: &Board) -> ToSgf<'a, C, E>
    where C: Param, E: Value
{
    ToSgf {
        value_type: ::std::marker::PhantomData::<C>,
        starting_point: starting_point.clone(),
        root: &root
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
pub struct GreedyPath<'a, E: Value + 'a> {
    current: &'a Node<E>,
}

impl<'a, E: Value + 'a> Iterator for GreedyPath<'a, E> {
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

/// Type alias for `Node<E>` that acts as a wrapper for calling `as_sgf` from
/// within a `write!` macro.
pub struct ToPretty<'a, E: Value + 'a> {
    root: &'a Node<E>,
}

impl<'a, E: Value + 'a> fmt::Display for ToPretty<'a, E> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let is_additional = (0..362).any(|i| self.root.amaf_count[i] > 0);
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
            let additional = if is_additional {
                format!("(A: {:5.2}%: {:7}) ",
                    100.0 * self.root.amaf[i],
                    self.root.amaf_count[i],
                )
            } else {
                String::new()
            };

            write!(fmt, "{: >5} -> {:7} (W: {:5.2}%) {}(N: {:5.2}%) PV: {} {}\n",
                pretty_vertex,
                child.total_count,
                100.0 * self.root.value[i],
                additional,
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
pub fn to_pretty<'a, E: Value>(root: &'a Node<E>) -> ToPretty<'a, E> {
    ToPretty { root: root }
}

#[cfg(test)]
mod tests {
    use test::{self, Bencher};
    use rand::{XorShiftRng, Rng};
    use go::*;
    use mcts::param::*;
    use mcts::tree::*;

    #[bench]
    fn argmax_each(b: &mut Bencher) {
        let mut array = [::std::f32::NEG_INFINITY; 368];

        // test setting each element within an eigth lane as the maximum to
        // ensure nothing is lost
        for i in 0..362 {
            array[i] = 2.0 + (i as f32);

            assert_eq!(argmax(&array), Some(i));
        }

        let array = test::black_box(array);

        b.iter(move || {
            argmax(&array)
        });
    }

    #[test]
    fn argmax_neg() {
        let mut array = [-1.0f32; 368];
        array[234] = -0.1;

        assert_eq!(argmax(&array), Some(234));
    }

    fn get_prior_distribution(rng: &mut Rng) -> Box<[f32]> {
        let mut prior: Vec<f32> = (0..362).map(|_| rng.next_f32()).collect();
        let sum_recip: f32 = prior.iter().map(|&f| f).sum();

        for i in 0..362 {
            prior[i] *= sum_recip;
        }

        prior.into_boxed_slice()
    }

    unsafe fn bench_test<C: Param, E: Value>(b: &mut Bencher) {
        let mut rng = XorShiftRng::new_unseeded();
        let mut root = Node::<E>::new(Color::Black, get_prior_distribution(&mut rng));

        for t in 0..800 {
            let mut board = Board::new();
            let mut dst_ref = [0.0f32; 368];
            let mut dst_asm = [0.0f32; 368];

            // check so that the reference and asm implementation gives back the same
            // value
            E::get::<C, E>(&root, &mut dst_asm);
            E::get_ref::<C, E>(&root, &mut dst_ref);

            for i in 0..362 {
                // because of numeric instabilities and approximations the answers may
                // be slightly different
                const EPS: f32 = 1e-1;

                assert!(
                    (dst_asm[i] - dst_ref[i]).abs() < EPS,
                    "epoch {}: dst[{}] <=> asm {} != ref {}",
                    t, i, dst_asm[i], dst_ref[i]
                );
            }

            // expand the tree by one probe so that the root values change
            let trace = probe::<C, E>(&mut root, &mut board).unwrap();
            let &(_, color, _) = trace.last().unwrap();
            let next_color = color.opposite();
            let (value, policy) = (rng.next_f32(), get_prior_distribution(&mut rng));

            insert::<C, E>(&trace, next_color, value, policy);
        }

        // benchmark the value function only
        let root = test::black_box(root);

        b.iter(|| {
            let mut dst = test::black_box([0.0f32; 368]);

            E::get::<C, E>(&root, &mut dst);
            dst
        });

    }

    #[bench]
    fn puct(b: &mut Bencher) {
        unsafe { bench_test::<Standard, PUCT>(b); }
    }

    #[bench]
    fn rave(b: &mut Bencher) {
        unsafe { bench_test::<Standard, RAVE>(b); }
    }
}