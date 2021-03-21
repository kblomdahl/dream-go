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

use dg_go::utils::sgf::SgfCoordinate;
use dg_go::{Board, Color, Point};
use dg_utils::lcb::normal_lcb_m;
use dg_utils::{config, max};
use super::asm::{argmax_f32, argmax_i32};
use super::choose::choose;
use super::parallel::spin::Mutex;
use super::parallel::global_rwlock;

use ordered_float::OrderedFloat;
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::fmt;
use std::mem::ManuallyDrop;
use std::intrinsics::{atomic_xadd, atomic_xsub, atomic_cxchg};
use std::ptr;

/// The minimum number of visits to use the lower-confidence bound instead
/// of number of probes during search to pick the "best" child.
const MIN_LCB_VISITS: i32 = 80;

/// An implementation of the selection heuristic as suggested in the AlphaGo
/// Zero paper [1].
///
/// [1] https://www.nature.com/articles/nature24270
#[derive(Clone)]
pub struct UCT;

impl UCT {
    #[inline(always)]
    unsafe fn get_impl<C: Children>(node: &Node, child: &C, value: &mut [f32]) {
        use std::intrinsics::{fadd_fast, fdiv_fast, fmul_fast};

        let n = node.total_count + node.vtotal_count;
        let sqrt_n = ((1 + n) as f32).sqrt();
        let uct_exp = config::get_uct_exp(n);
        let uct_exp_sqrt_n = fmul_fast(uct_exp, sqrt_n);

        for i in 0..362 {
            let count = child.total_count(i);
            let prior = node.prior[i];
            let value_ = value[i];
            let exp_bonus = if count == 0 { uct_exp_sqrt_n } else { fdiv_fast(uct_exp_sqrt_n, (1 + count) as f32) };

            value[i] = fadd_fast(value_, fmul_fast(prior, exp_bonus));
        }
    }

    #[allow(unused_attributes)]
    #[target_feature(enable = "avx,avx2")]
    unsafe fn get_avx2<C: Children>(node: &Node, child: &C, value: &mut [f32]) {
        UCT::get_impl(node, child, value);
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
        use std::intrinsics::{fadd_fast, fsub_fast, fdiv_fast, fmul_fast};

        for &(node, _, index) in trace.iter() {
            let value_ = if color == (*node).to_move { value } else { 1.0 - value };

            // incremental update of the average value and remove any additional
            // virtual losses we added to the node
            atomic_xadd(&mut (*node).total_count, 1);
            atomic_xsub(&mut (*node).vtotal_count, *config::VLOSS_CNT);

            (*node).children.with_mut(index, |mut child| {
                let _guard = (*node).lock.lock();

                let prev_value = child.value();
                let prev_value_s = child.value_s();
                let prev_count = child.add_count(1);
                let next_value = child.set_value(fadd_fast(
                    prev_value,
                    fdiv_fast(
                        fsub_fast(value_, prev_value),
                        (prev_count + 1) as f32
                    )
                ));
                child.set_value_s(fadd_fast(
                    prev_value_s,
                    fmul_fast(
                        fsub_fast(value_, prev_value),
                        fsub_fast(value_, next_value)
                    )
                ));
                child.sub_vcount(*config::VLOSS_CNT);
            }, (*node).initial_value);
        }
    }

    /// Optimized implementation of the PUCT value function.
    ///
    /// # Arguments
    ///
    /// * `node` -
    /// * `value` - the winrates to use in the calculations
    ///
    #[inline(always)]
    fn get(node: &Node, value: &mut [f32]) {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                match node.children {
                    ChildrenImpl::Small(ref small) => UCT::get_avx2(node, &**small, value),
                    ChildrenImpl::Big(ref big) => UCT::get_avx2(node, &**big, value),
                }
            }
        } else {
            unsafe {
                match node.children {
                    ChildrenImpl::Small(ref small) => UCT::get_impl(node, &**small, value),
                    ChildrenImpl::Big(ref big) => UCT::get_impl(node, &**big, value),
                }
            }
        }
    }
}

/// An implementation of the First-Play Urgency.
pub struct FPU;

impl FPU {
    unsafe fn apply_impl<C: Children>(value: &mut [f32], child: &C, fpu_reduce: f32) {
        use std::intrinsics::fsub_fast;

        for i in 0..368 {
            let count = child.total_count(i);

            if count == 0 {
                value[i] = max(0.0, fsub_fast(value[i], fpu_reduce));
            }
        }
    }

    #[target_feature(enable = "avx,avx2")]
    unsafe fn apply_avx2<C: Children>(value: &mut [f32], child: &C, fpu_reduce: f32) {
        FPU::apply_impl(value, child, fpu_reduce)
    }

    /// Apply the first play urgency reduction to all elements in `value` if `count`
    /// is zero.
    ///
    /// # Arguments
    ///
    /// * `value` - the value of each element
    /// * `child` - The children storage
    /// * `fpu_reduce` - the reduction to apply
    ///
    #[inline(always)]
    pub fn apply<C: Children>(value: &mut [f32], child: &C, fpu_reduce: f32) {
        if is_x86_feature_detected!("avx2") {
            unsafe { FPU::apply_avx2(value, child, fpu_reduce) }
        } else {
            unsafe { FPU::apply_impl(value, child, fpu_reduce) }
        }
    }
}

/// Flyweight structure used to contain the values of a single child in a `Node`. These
/// values should never be modified as they will **not** be synchronized back to the
/// origin structure.
pub struct Child {
    expanding: bool,
    count: i32,
    vcount: i16,
    ptr: *mut Node,
    value: f32,
    value_s: f32
}

impl Child {
    /// Returns a default child, with the given `value`. This constructor is normally used
    /// when a sparse `Node` does not contain a child it was asked for.
    ///
    /// # Arguments
    ///
    /// * `value` - the value of the parent `Node`
    ///
    fn with_value(value: f32) -> Child {
        Child {
            count: 0,
            vcount: 0,
            value: value,
            value_s: 0.0,
            expanding: false,
            ptr: ptr::null_mut()
        }
    }

    /// Returns a child that is initialized from a `SmallChildrenImpl` at the given `index`.
    ///
    /// # Arguments
    ///
    /// * `small` - the `SmallChildrenImpl` to initialize from
    /// * `index` - the sparse index in `SmallChildrenImpl` to initialize from
    ///
    fn from_small(small: &SmallChildrenImpl, index: usize) -> Child {
        Child {
            count: small.count[index],
            vcount: small.vcount[index],
            value: small.value[index],
            value_s: small.value_s[index],
            expanding: small.expanding[index],
            ptr: small.ptr[index]
        }
    }

    /// Returns a child that is initialized from a `BigChildrenImpl` at the given `index`.
    ///
    /// # Arguments
    ///
    /// * `big` - the `BigChildrenImpl` to initialize from
    /// * `index` - the dense index in `BigChildrenImpl` to initialize from
    ///
    fn from_big(big: &BigChildrenImpl, index: usize) -> Child {
        debug_assert!(index < 362, "{}", index);

        Child {
            count: big.count[index],
            vcount: big.vcount[index],
            value: big.value[index],
            value_s: big.value_s[index],
            expanding: big.expanding[index],
            ptr: big.ptr[index]
        }
    }

    /// Returns whether this child is currently being expanded.
    pub fn expanding(&self) -> bool {
        self.expanding
    }

    /// Returns the number of visits to this child.
    pub fn count(&self) -> i32 {
        self.count
    }

    /// Returns the number of virtual visits to this child.
    pub fn vcount(&self) -> i32 {
        self.vcount as i32
    }

    /// Return the child node itself.
    pub fn ptr(&self) -> *mut Node {
        self.ptr
    }

    /// Returns the average value of this child.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Returns the variance of the value of this child.
    pub fn value_var(&self) -> f32 {
        self.value_s / (self.count as f32 + 1e-5)
    }

    /// Returns the standard deviation of the value of this child.
    pub fn value_std(&self) -> f32 {
        self.value_var().sqrt()
    }
}

/// Flyweight mutable structure used to contain the values of a single child in a `Node`.
#[derive(Debug)]
pub struct ChildMut {
    expanding: *mut bool,
    count: *mut i32,
    vcount: *mut i16,
    ptr: *mut *mut Node,
    value: *mut f32,
    value_s: *mut f32
}

impl ChildMut {
    /// Returns a child that is initialized from a `SmallChildrenImpl` at the given `index`. The
    /// given `small` node must outlive the returned `ChildMut`.
    ///
    /// # Arguments
    ///
    /// * `small` - the `SmallChildrenImpl` to initialize from
    /// * `index` - the sparse index in `SmallChildrenImpl` to initialize from
    ///
    unsafe fn from_small(small: &mut SmallChildrenImpl, index: usize) -> ChildMut {
        ChildMut {
            count: small.count.get_unchecked_mut(index),
            vcount: small.vcount.get_unchecked_mut(index),
            value: small.value.get_unchecked_mut(index),
            value_s: small.value_s.get_unchecked_mut(index),
            expanding: small.expanding.get_unchecked_mut(index),
            ptr: small.ptr.get_unchecked_mut(index)
        }
    }

    /// Returns a child that is initialized from a `BigChildrenImpl` at the given `index`. The
    /// given `big` node must outlive the returned `ChildMut`.
    ///
    /// # Arguments
    ///
    /// * `big` - the `BigChildrenImpl` to initialize from
    /// * `index` - the dense index in `BigChildrenImpl` to initialize from
    ///
    unsafe fn from_big(big: &mut BigChildrenImpl, index: usize) -> ChildMut {
        debug_assert!(index < 362);

        ChildMut {
            count: big.count.get_unchecked_mut(index),
            vcount: big.vcount.get_unchecked_mut(index),
            value: big.value.get_unchecked_mut(index),
            value_s: big.value_s.get_unchecked_mut(index),
            expanding: big.expanding.get_unchecked_mut(index),
            ptr: big.ptr.get_unchecked_mut(index)
        }
    }

    /// Returns whether this child is currently being expanded.
    pub fn expanding(&self) -> bool {
        unsafe { *self.expanding }
    }

    /// Returns the number of visits to this child.
    pub fn count(&self) -> i32 {
        unsafe { *self.count }
    }

    /// Returns the number of virtual visits to this child.
    pub fn vcount(&self) -> i32 {
        unsafe { *self.vcount as i32 }
    }

    /// Return the child node itself.
    pub fn ptr(&self) -> *mut Node {
        unsafe { *self.ptr }
    }

    /// Returns the average value of this child.
    pub fn value(&self) -> f32 {
        unsafe { *self.value }
    }

    /// Returns the average value of this child.
    pub fn value_s(&self) -> f32 {
        unsafe { *self.value_s }
    }

    /// Sets this child as having been expanded, returning whether this
    /// has occurred before.
    fn set_expanding(&mut self) -> bool {
        use std::intrinsics::atomic_or;

        unsafe { atomic_or(self.expanding as *mut u8, true as u8) != 0 }
    }

    /// Unsets this child as having been expanding.
    fn unset_expanding(&mut self) {
        use std::intrinsics::atomic_and;

        unsafe { atomic_and(self.expanding as *mut u8, false as u8) };
    }

    /// Sets the number of visits to this child.
    ///
    /// # Arguments
    ///
    /// * `value` - the new number of visits to this child
    ///
    fn set_count(&mut self, value: i32) {
        unsafe { *self.count = value; }
    }

    /// Increments the number of visits to this child. Returning the previous
    /// value.
    ///
    /// # Arguments
    ///
    /// * `count` - the number of visits to add
    ///
    fn add_count(&mut self, count: i32) -> i32 {
        unsafe { atomic_xadd(self.count, count) }
    }

    /// Increments the number of virtual visits to this child. Returning
    /// the previous value.
    ///
    /// # Arguments
    ///
    /// * `count` - the number of virtual visits to add
    ///
    fn add_vcount(&mut self, count: i32) -> i16 {
        unsafe { atomic_xadd(self.vcount, count as i16) }
    }

    /// Decrements the number of virtual visits to this child. Returning
    /// the previous value.
    ///
    /// # Arguments
    ///
    /// * `count` - the number of virtual visits to remove
    ///
    fn sub_vcount(&mut self, count: i32) -> i16 {
        unsafe { atomic_xsub(self.vcount, count as i16) }
    }

    /// Sets the actual child node. If there is already a child node set, then you
    /// are responsible for freeing the old node.
    ///
    /// # Arguments
    ///
    /// * `value` - the new child `Node`
    ///
    fn set_ptr(&mut self, value: *mut Node) {
        unsafe { *self.ptr = value; }
    }

    /// Sets the average value of this child.
    ///
    /// # Arguments
    ///
    /// * `value` - the new average value of this child
    ///
    fn set_value(&mut self, value: f32) -> f32 {
        unsafe { *self.value = value; }

        value
    }

    /// Sets the square sum of average distances for the value of this child.
    ///
    /// # Arguments
    ///
    /// * `value_s` - new square sum of this child
    ///
    fn set_value_s(&mut self, value_s: f32) {
        unsafe { *self.value_s = value_s; }
    }
}

pub trait Children {
    fn count(&self, index: usize) -> i32;

    fn virtual_count(&self, index: usize) -> i32;

    fn total_count(&self, index: usize) -> i32 {
        self.count(index) + self.virtual_count(index)
    }
}

/// A dense representation of a `Node`.
#[repr(align(64))]
pub struct BigChildrenImpl {
    /// The number of times each edge has been traversed.
    pub count: [i32; 368],

    /// The number of virtual losses each edge has.
    pub vcount: [i16; 368],

    /// The average value for the sub-tree of each edge.
    pub value: [f32; 368],

    /// The sum of squares of the value for the sub-tree of each edge. This can be used to
    /// calculate the variance of the value by `(s / count).sqrt()`.
    pub value_s: [f32; 368],

    /// Whether some thread is currently busy (or is done) expanding the given
    /// child. This is used to avoid the same child being expanded multiple
    /// times by different threads.
    expanding: [bool; 362],

    /// The sub-tree that each edge points towards.
    ptr: [*mut Node; 362]
}

impl Children for BigChildrenImpl {
    fn count(&self, index: usize) -> i32 {
        self.count[index]
    }

    fn virtual_count(&self, index: usize) -> i32 {
        self.vcount[index] as i32
    }
}

impl Drop for BigChildrenImpl {
    fn drop(&mut self) {
        for &child in self.ptr.iter() {
            if !child.is_null() {
                unsafe { Box::from_raw(child); }
            }
        }
    }
}

impl BigChildrenImpl {
    /// Returns a `BigChildrenImpl` that is equivalent to the given `small` node.
    ///
    /// # Arguments
    ///
    /// * `small` - the node to initialize from
    /// * `value` - the initial _value_ to use for any children not in `small`
    ///
    unsafe fn from_small(small: &SmallChildrenImpl, value: f32) -> Self {
        let mut big = Self {
            count: [0; 368],
            vcount: [0; 368],
            value: [value; 368],
            value_s: [0.0; 368],
            expanding: [false; 362],
            ptr: [ptr::null_mut(); 362]
        };

        for (index, &other) in small.indices.iter().enumerate() {
            if other >= 0 {
                let other = other as usize;

                big.count[other] = small.count[index];
                big.vcount[other] = small.vcount[index];
                big.value[other] = small.value[index];
                big.value_s[other] = small.value_s[index];
                big.expanding[other] = small.expanding[index];
                big.ptr[other] = small.ptr[index];
            }
        }

        // set the meta information of moves that are out of bounds to -Inf to
        // ensure that they are never picked
        for i in 362..368 {
            big.count[i] = ::std::i32::MIN;
            big.value[i] = ::std::f32::NEG_INFINITY;
        }

        big
    }
}

/// The maximum number of elements a sparse (small) node contains.
const SMALL_SIZE: usize = 8;

/// Possible results for looking up an index in a small node.
enum SmallChildrenResult {
    Found(usize),
    NotFound(usize),
    Overflow
}

/// A sparse representation of a `Node` that only stores `SMALL_SIZE` children before
/// overflowing. It store the sparse indices in `indices`, which is an unsorted mapping
/// from sparse index to dense index.
pub struct SmallChildrenImpl {
    /// The number of times each edge has been traversed.
    pub count: [i32; SMALL_SIZE],

    /// The number of virtual losses each edge has.
    pub vcount: [i16; SMALL_SIZE],

    /// The average value for the sub-tree of each edge.
    pub value: [f32; SMALL_SIZE],

    /// The sum of squares of the value for the sub-tree of each edge. This can be used to
    /// calculate the variance of the value by `(s / count).sqrt()`.
    pub value_s: [f32; SMALL_SIZE],

    /// Whether some thread is currently busy (or is done) expanding the given
    /// child. This is used to avoid the same child being expanded multiple
    /// times by different threads.
    expanding: [bool; SMALL_SIZE],

    /// The sub-tree that each edge points towards.
    ptr: [*mut Node; SMALL_SIZE],

    /// Indices of the children stored in this node.
    indices: [i16; SMALL_SIZE]
}

impl Children for SmallChildrenImpl {
    fn count(&self, index: usize) -> i32 {
        if let Ok(i) = unsafe { self.find_index_fast(index) } {
            self.count[i]
        } else {
            0
        }
    }

    fn virtual_count(&self, index: usize) -> i32 {
        if let Ok(i) = unsafe { self.find_index_fast(index) } {
            self.vcount[i] as i32
        } else {
            0
        }
    }

    fn total_count(&self, index: usize) -> i32 {
        if let Ok(i) = unsafe { self.find_index_fast(index) } {
            self.count[i] + self.vcount[i] as i32
        } else {
            0
        }
    }
}

impl Drop for SmallChildrenImpl {
    fn drop(&mut self) {
        for &child in self.ptr.iter() {
            if !child.is_null() {
                unsafe { Box::from_raw(child); }
            }
        }
    }
}

impl SmallChildrenImpl {
    /// Returns an empty sparse node.
    ///
    /// # Arguments
    ///
    /// * `value` - the initial _value_ for any child
    ///
    fn with_value(value: f32) -> Self {
        Self {
            count: [0; SMALL_SIZE],
            vcount: [0; SMALL_SIZE],
            value: [value; SMALL_SIZE],
            value_s: [0.0; SMALL_SIZE],
            expanding: [false; SMALL_SIZE],
            ptr: [ptr::null_mut(); SMALL_SIZE],
            indices: [::std::i16::MIN; SMALL_SIZE]
        }
    }

    /// Returns the sparse index for the given dense `index`, or the insertion index if it
    /// not exist in this node.
    ///
    /// # Arguments
    ///
    /// * `index` - the index to search for
    ///
    #[inline(always)]
    unsafe fn find_index_fast(&self, index: usize) -> Result<usize, usize> {
        use std::arch::x86_64::*;

        let indices = _mm_loadu_si128(&self.indices as *const i16 as *const _);
        let eq = _mm_cmpeq_epi16(indices, _mm_set1_epi16(index as i16));
        let eq = _mm_movemask_epi8(eq) as u32;

        if eq != 0 {
            let trailing_zeros = _mm_tzcnt_32(eq) as usize;

            Ok(trailing_zeros / 2)
        } else {
            // find the first element _< 0_
            let lt = _mm_cmplt_epi16(indices, _mm_set1_epi16(0));
            let lt = _mm_movemask_epi8(lt) as u32;

            if lt != 0 {
                Err(_mm_tzcnt_32(lt) as usize / 2)
            } else {
                Err(::std::usize::MAX)
            }
        }
    }

    /// Returns the sparse index for the given dense `index`, or the index where it
    /// can be inserted.
    ///
    /// # Arguments
    ///
    /// * `index` - the index to search for
    ///
    fn find_index(&self, index: usize) -> SmallChildrenResult {
        match unsafe { self.find_index_fast(index) } {
            Ok(index) => SmallChildrenResult::Found(index),
            Err(other) if other == ::std::usize::MAX => { SmallChildrenResult::Overflow },
            Err(other) => { SmallChildrenResult::NotFound(other) }
        }
    }
}

/// Iterator over any non-zero child in a sparse node.
pub struct ChildrenNonZeroIter {
    count: *const i32,
    indices: *const i16,
    index: usize,
    len: usize
}

impl Iterator for ChildrenNonZeroIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        unsafe {
            while self.index < self.len {
                let prev_index = self.index;
                self.index += 1;

                if *self.count.add(prev_index) > 0 {
                    return Some(if self.indices.is_null() {
                        prev_index
                    } else {
                        *self.indices.add(prev_index) as usize
                    });
                }
            }

            None
        }
    }
}

/// Union of `SmallChildrenImpl` and `BigChildrenImpl`, where the later is stored on the heap.
pub enum ChildrenImpl {
    Small(ManuallyDrop<SmallChildrenImpl>),
    Big(Box<BigChildrenImpl>)
}

unsafe impl Send for ChildrenImpl {}

impl ChildrenImpl {
    /// Returns the scattered value array of all children.
    ///
    /// # Arguments
    ///
    /// * `default_value` - the value to use for any unvisited children
    ///
    pub fn value(&self, default_value: f32) -> Vec<f32> {
        match *self {
            ChildrenImpl::Big(ref big) => {
                big.value.to_vec()
            },
            ChildrenImpl::Small(ref small) => {
                let mut out = vec! [default_value; 368];

                for index in 0..SMALL_SIZE {
                    let other = small.indices[index];

                    if other >= 0 {
                        out[other as usize] = small.value[index];
                    }
                }

                out
            }
        }
    }

    /// Returns the index of the child with the largest number of visits.
    pub fn argmax_count(&self) -> usize {
        match *self {
            ChildrenImpl::Big(ref big) => argmax_i32(&big.count).unwrap(),
            ChildrenImpl::Small(ref small) => {
                let other = argmax_i32(&small.count).unwrap();
                let index = small.indices[other];

                if index < 0 {
                    0
                } else {
                    index as usize
                }
            }
        }
    }

    /// Returns the index of the child with the largest average value.
    pub fn argmax_value(&self) -> usize {
        match *self {
            ChildrenImpl::Big(ref big) => argmax_f32(&big.value).unwrap(),
            ChildrenImpl::Small(ref small) => {
                let other = argmax_f32(&small.value).unwrap();

                small.indices[other] as usize
            }
        }
    }

    /// Returns an iterator over all visited children.
    pub fn nonzero(&self) -> ChildrenNonZeroIter {
        match self {
            ChildrenImpl::Small(ref small) => {
                ChildrenNonZeroIter {
                    count: &small.count as *const i32,
                    indices: &small.indices as *const i16,
                    index: 0,
                    len: SMALL_SIZE
                }
            },
            ChildrenImpl::Big(ref big) => {
                ChildrenNonZeroIter {
                    count: &big.count as *const i32,
                    indices: ptr::null(),
                    index: 0,
                    len: 362
                }
            }
        }
    }

    /// Returns the result of the given callback, and being called with an immutable
    /// reference for the child for index.
    ///
    /// # Arguments
    ///
    /// * `index` -
    /// * `callback` -
    /// * `initial_value` -
    ///
    pub fn with<T, F>(&self, index: usize, callback: F, initial_value: f32) -> T
        where F: FnOnce(Child) -> T
    {
        callback(match self {
            ChildrenImpl::Small(ref small) => {
                match small.find_index(index) {
                    SmallChildrenResult::Found(other) => {
                        Child::from_small(small, other)
                    },
                    _ => {
                        Child::with_value(initial_value)
                    }
                }
            },
            ChildrenImpl::Big(ref big) => {
                Child::from_big(big, index)
            }
        })
    }

    /// Returns a `ChildMut` for the given child in the _small_ node implementation, or _None_ if
    /// the implementation is full and needs to be extended.
    ///
    /// # Arguments
    ///
    /// * `small` - the children implementation
    /// * `index` - the index to fetch
    ///
    fn with_mut_small(small: &mut SmallChildrenImpl, index: usize) -> Option<ChildMut> {
        'retry: loop {
            return match small.find_index(index) {
                SmallChildrenResult::Found(other) => {
                    Some(unsafe { ChildMut::from_small(small, other) })
                },
                SmallChildrenResult::NotFound(other) => {
                    unsafe {
                        let indices_other = small.indices.as_mut_ptr().add(other);

                        if atomic_cxchg(indices_other, ::std::i16::MIN, index as i16) != (::std::i16::MIN, true) {
                            continue 'retry;
                        }
                    }

                    Some(unsafe { ChildMut::from_small(small, other) })
                },
                SmallChildrenResult::Overflow => {
                    None
                }
            }
        }
    }

    /// Returns the result of the given callback, and being called with an mutable
    /// reference for the child for index.
    ///
    /// # Arguments
    ///
    /// * `index` -
    /// * `callback` -
    /// * `initial_value` -
    ///
    pub fn with_mut<T, F>(&mut self, index: usize, callback: F, initial_value: f32) -> T
        where F: FnOnce(ChildMut) -> T
    {
        let child = match self {
            ChildrenImpl::Small(ref mut small) => {
                ChildrenImpl::with_mut_small(small, index)
            },
            ChildrenImpl::Big(ref mut big) => {
                Some(unsafe { ChildMut::from_big(big, index) })
            }
        };

        if let Some(child) = child {
            callback(child)
        } else {
            global_rwlock::write(|| {
                unsafe {
                    *self = ChildrenImpl::Big(match self {
                        ChildrenImpl::Big(ref _big) => { return },
                        ChildrenImpl::Small(ref small) => {
                            Box::new(BigChildrenImpl::from_small(small, initial_value))
                        }
                    });
                }
            });

            self.with_mut(index, callback, initial_value)
        }
    }
}

/// The result of selecting which is the next move:
///
/// * `Conflict` - We tried to expand the same move as someone else.
/// * `NoResult` - There were no valid moves to consider.
/// * `Found` -
///
pub enum ProbeResult<T> {
    Conflict,
    NoResult,
    Found(T)
}

impl<T> ProbeResult<T> {
    pub fn is_some(&self) -> bool {
        match *self {
            ProbeResult::Found(_) => true,
            _ => false
        }
    }

    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    pub fn unwrap(self) -> T {
        match self {
            ProbeResult::Found(x) => x,
            _ => panic!()
        }
    }
}

/// A monte carlo search tree.
#[repr(align(64))]
pub struct Node {
    /// Spinlock used to protect the data in this node during modifications.
    pub lock: Mutex,

    /// The color of each edge.
    pub to_move: Color,

    /// The initial vale of this node.
    pub initial_value: f32,

    /// The number of consecutive passes to reach this node.
    pub pass_count: i16,

    /// The total number of times any edge has been traversed.
    pub total_count: i32,

    /// The total number of virtual losses for any edge.
    pub vtotal_count: i32,

    /// The prior value of each edge as indicated by the policy.
    pub prior: [f32; 368],

    /// The sparse (or dense) representation of the remaining MCTS fields.
    pub children: ChildrenImpl
}

impl Drop for Node {
    fn drop(&mut self) {
        if let ChildrenImpl::Small(ref mut small) = self.children {
            unsafe { ManuallyDrop::drop(small) }
        }
    }
}

impl Node {
    /// Returns an empty search tree with the given starting color and prior
    /// values.
    ///
    /// # Arguments
    ///
    /// * `to_move` - the color of the first players color
    /// * `prior` - the prior values of the nodes
    ///
    pub fn new(to_move: Color, value: f32, prior: Vec<f32>) -> Node {
        assert!(prior.len() >= 362);

        // copy the prior values into an array size that is dividable
        // by 16 to ensure we can use 256-bit wide SIMD registers.
        let mut prior_padding = [::std::f32::NEG_INFINITY; 368];
        prior_padding[..362].copy_from_slice(&prior[..362]);

        Node {
            lock: Mutex::new(),
            to_move: to_move,
            initial_value: value,
            pass_count: 0,
            total_count: 0,
            vtotal_count: 0,
            prior: prior_padding,
            children: ChildrenImpl::Small(ManuallyDrop::new(SmallChildrenImpl::with_value(value)))
        }
    }

    /// Returns true if the given vertex is a valid candidate move in this tree.
    ///
    /// # Arguments
    ///
    /// * `board` -
    /// * `index` -
    ///
    fn is_valid_candidate(&self, board: &Board, index: usize) -> bool {
        self.prior[index].is_finite() && {
            index == 361 || board.is_valid(self.to_move, Point::from_packed_parts(index))
        } && self.with(index, |cand| cand.value().is_finite())
    }

    /// Returns true if any of the valid candidate moves in this tree are
    /// valid moves on the given board.
    ///
    /// # Arguments
    ///
    /// * `board` -
    ///
    pub fn has_valid_candidates(&self, board: &Board) -> bool {
        (0..362).any(|i| self.is_valid_candidate(board, i))
    }

    /// Returns the total size of this search tree.
    pub fn size(&self) -> usize {
        self.total_count as usize
    }

    /// Returns the result of the given callback, and being called with an immutable
    /// reference for the child for index.
    ///
    /// # Arguments
    ///
    /// * `index` -
    /// * `callback` -
    ///
    pub fn with<T, F>(&self, index: usize, callback: F) -> T
        where F: FnOnce(Child) -> T
    {
        self.children.with(index, callback, self.initial_value)
    }

    /// Returns the result of the given callback, and being called with an mutable
    /// reference for the child for index.
    ///
    /// # Arguments
    ///
    /// * `index` -
    /// * `callback` -
    ///
    pub fn with_mut<T, F>(&mut self, index: usize, callback: F) -> T
        where F: FnOnce(ChildMut) -> T
    {
        self.children.with_mut(index, callback, self.initial_value)
    }

    fn as_sgf<S: SgfCoordinate>(&self, fmt: &mut fmt::Formatter, meta: bool) -> fmt::Result {
        // annotate the top-10 moves to make it easier to navigate for the
        // user.
        let mut children = (0..362).collect::<Vec<usize>>();
        children.sort_by_key(|&i| -self.with(i, |child| child.count()));

        if meta {
            for i in 0..10 {
                let j = children[i];

                if j != 361 && self.with(j, |child| child.count()) > 0 {
                    lazy_static! {
                        static ref LABELS: Vec<&'static str> = vec! [
                            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
                        ];
                    }

                    write!(fmt, "LB[{}:{}]",
                        S::to_sgf(Point::from_packed_parts(j)),
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

        let mut uct = self.children.value(self.initial_value);
        UCT::get(self, &mut uct);

        for i in children {
            // do not output nodes that has not been visited to reduce the
            // size of the final SGF file.
            if self.with(i, |child| child.count()) == 0 {
                continue;
            }

            write!(fmt, "(")?;
            write!(fmt, ";{}[{}]",
                   if self.to_move == Color::Black { "B" } else { "W" },
                   if i == 361 { "tt".to_string() } else { S::to_sgf(Point::from_packed_parts(i)) }
            )?;
            write!(fmt, "C[prior {:.4} value {:.4} (visits {} / total {}) uct {:.4}]",
                self.prior[i],
                self.with(i, |child| child.value()),
                self.with(i, |child| child.count()),
                self.total_count,
                uct[i]
            )?;

            unsafe {
                let child = self.with(i, |child| child.ptr());

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
    /// # Arguments
    ///
    /// * `self` - the search tree to pluck the child from
    /// * `index` - the move to pluck the sub-tree for
    ///
    pub fn forward(mut self, index: usize) -> Option<Node> {
        let color = self.to_move;
        let pass_count = self.pass_count;

        self.with_mut(index, |mut child| {
            if child.ptr().is_null() {
                if index == 361 {
                    // we need to record that were was a pass so that we have the correct
                    // pass count in the root node.
                    let prior = vec! [0.0f32; 362];
                    let mut next = Node::new(color.opposite(), 0.5, prior);
                    next.pass_count = pass_count + 1;

                    Some(next)
                } else {
                    None
                }
            } else {
                let next = child.ptr();
                child.set_ptr(ptr::null_mut());

                Some(unsafe { ptr::read(next) })
            }
        })
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
            let max_i = self.children.nonzero()
                .max_by(|&a, &b| compare_children(self, a, b, MIN_LCB_VISITS))
                .unwrap_or(361);

            (self.with(max_i, |child| child.value()), max_i)
        } else {
            let visits = (0..362)
                .map(|i| self.with(i, |child| child.count()))
                .collect::<Vec<i32>>();
            let temperature = (temperature as f64).recip();
            let at = thread_rng().gen::<f64>();

            if let Some((i, _)) = choose(&visits, 0.5, temperature, at) {
                (self.with(i, |child| child.value()), i)
            } else {
                (self.initial_value, 361)  // no valid moves
            }
        }
    }

    /// Returns the best move according to the prior value of the root node.
    pub fn prior(&self) -> (f32, usize) {
        let max_i = argmax_f32(&self.prior).unwrap_or(361);

        (self.prior[max_i], max_i)
    }

    /// Returns a vector containing the _correct_ normalized probability that each move
    /// should be played given the current search tree.
    pub fn softmax<T: From<f32> + Clone>(&self) -> Vec<T> {
        let mut s = vec! [T::from(0.0f32); 362];
        let mut s_total = 0.0f32;

        for i in self.children.nonzero() {
            s_total += self.with(i, |child| child.count()) as f32;
        }

        for i in self.children.nonzero() {
            s[i] = T::from(self.with(i, |child| child.count()) as f32 / s_total);
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
        self.with_mut(index, |mut child| {
            child.set_value(::std::f32::NEG_INFINITY);
            child.set_count(0);
        });
    }

    /// Returns the child with the maximum UCT value, and increase its visit count
    /// by one.
    ///
    /// # Arguments
    ///
    /// * `apply_fpu` - whether to use the first-play urgency heuristic
    ///
    fn select(&mut self, apply_fpu: bool) -> ProbeResult<(usize, f32)> {
        let mut value = self.children.value(self.initial_value);

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
            let fpu_reduce = config::get_fpu_reduce(self.total_count + self.vtotal_count);

            match self.children {
                ChildrenImpl::Big(ref big) => FPU::apply(&mut value, &**big, fpu_reduce),
                ChildrenImpl::Small(ref small) => FPU::apply(&mut value, &**small, fpu_reduce)
            }
        }

        // compute all UCB1 values for each node before trying to figure out which
        // to pick to make it possible to do it with SIMD.
        for i in 362..368 {
            value[i] = ::std::f32::NEG_INFINITY;
        }

        UCT::get(self, &mut value);

        // greedy selection based on the maximum ucb1 value, failing if someone else
        // is already expanding the node we want to expand.
        let initial_value = self.initial_value;
        let max_i = argmax_f32(&value);
        let max_i =
            if let Some(i) = max_i {
                self.children.with_mut(i, |mut child| {
                    if child.set_expanding() && child.ptr().is_null() {
                        ProbeResult::Conflict
                    } else {
                        child.add_vcount(*config::VLOSS_CNT);

                        ProbeResult::Found((i, value[i]))
                    }
                }, initial_value)
            } else {
                ProbeResult::NoResult
            };

        if let ProbeResult::Found(_) = max_i {
            unsafe {
                atomic_xadd(&mut self.vtotal_count, *config::VLOSS_CNT);
            }
        }

        max_i
    }
}

pub type NodeTrace = Vec<(*mut Node, Color, usize)>;

/// Undo a probe into the search tree by undoing any virtual losses, and / or visits
/// added during the probe.
///
/// # Arguments
///
/// * `trace` - the trace to undo
/// * `undo_expanding` - whether to also revert the `expanding` flag
///
pub unsafe fn undo(trace: NodeTrace, undo_expanding: bool) {
    for (node, _, next_child) in trace.into_iter() {
        atomic_xsub(&mut (*node).vtotal_count, *config::VLOSS_CNT as i32);

        (*node).children.with_mut(next_child, |mut child| {
            child.sub_vcount(*config::VLOSS_CNT);

            if undo_expanding && child.ptr().is_null() {
                child.unset_expanding();
            }
        }, (*node).initial_value);
    }
}

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
pub unsafe fn probe(root: &mut Node, board: &mut Board) -> ProbeResult<NodeTrace> {
    let mut trace = Vec::with_capacity(16);
    let mut current = root;

    loop {
        let apply_fpu = !trace.is_empty();

        match current.select(apply_fpu) {
            ProbeResult::Conflict => {
                undo(trace, false);
                return ProbeResult::Conflict;
            },
            ProbeResult::NoResult => {
                return ProbeResult::NoResult;
            },
            ProbeResult::Found((next_child, next_value)) => {
                trace.push((current as *mut Node, current.to_move, next_child));

                if next_child != 361 {  // not a passing move
                    let point = Point::from_packed_parts(next_child);

                    debug_assert!(
                        board.is_valid(current.to_move, point),
                        "{}\nnext_move {} {:?}, next_value {}\n{}",
                        board.to_string(),
                        current.to_move,
                        point,
                        next_value,
                        ToPretty { root: current, verbose: true }
                    );

                    board.place(current.to_move, point);
                } else if current.pass_count >= 1 {
                    break;  // at least two consecutive passes
                }

                //
                let child = current.with(next_child, |child| child.ptr());

                if child.is_null() {
                    break
                } else {
                    current = &mut *child;
                }
            }
        }
    }

    ProbeResult::Found(trace)
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
pub unsafe fn insert(trace: &NodeTrace, color: Color, value: f32, prior: Vec<f32>) {
    debug_assert!(value >= 0.0 && value <= 1.0);

    if let Some(&(node, _, index)) = trace.last() {
        let mut next = Box::new(Node::new(color, value, prior));
        if index == 361 {
            next.pass_count = (*node).pass_count + 1;
        }

        let updated = (*node).with_mut(index, |mut child| {
            if child.ptr().is_null() {
                child.set_ptr(Box::into_raw(next));
                true
            } else {
                false
            }
        });

        if !updated {
            debug_assert!(index == 361);

            // since we stop probing into a tree once two consecutive passes has
            // occurred we can double-expand those nodes. This is too prevent that
            // from causing memory leaks.
        }
    }

    UCT::update(trace, color, value);
}

/// Compare two children of an MCTS node such that the better candiate is bigger
/// than a worse candidate. The algorithm will compare the LCB if both
/// candidates has at least `min_lcb_visits` visit counts, otherwise fallback to
/// comparing the visit count.
///
/// # Arguments
///
/// * `node` -
/// * `a` -
/// * `b` -
/// * `min_lcb_visits` -
///
fn compare_children(
    node: &Node,
    a: usize,
    b: usize,
    min_lcb_visits: i32
) -> Ordering
{
    let a_count = node.with(a, |a| a.count());
    let b_count = node.with(b, |b| b.count());

    if a_count >= min_lcb_visits && b_count >= min_lcb_visits {
        let a_lcb = node.with(a, |a| normal_lcb_m(a.value(), a.value_std(), a.count(), node.total_count));
        let b_lcb = node.with(b, |b| normal_lcb_m(b.value(), b.value_std(), b.count(), node.total_count));

        if a_lcb != b_lcb {
            return OrderedFloat(a_lcb).cmp(&OrderedFloat(b_lcb));
        }
    }

    if a_count != b_count {
        return a_count.cmp(&b_count);
    }

    let a_prior = node.prior[a];
    let b_prior = node.prior[b];

    if a_prior != b_prior {
        return OrderedFloat(a_prior).cmp(&OrderedFloat(b_prior));
    }

    let a_value = node.with(a, |a| a.value());
    let b_value = node.with(b, |b| b.value());

    OrderedFloat(a_value).cmp(&OrderedFloat(b_value))
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
                if self.root.to_move == Color::Black { "B" } else { "W" }
            )?;

            // write the starting point to the SGF file as pre-set variables
            for point in Point::all() {
                match self.starting_point.at(point) {
                    None => Ok(()),
                    Some(Color::Black) => write!(fmt, "AB[{}]", S::to_sgf(point)),
                    Some(Color::White) => write!(fmt, "AW[{}]", S::to_sgf(point))
                }?
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
                LETTERS[Point::from_packed_parts(self.inner).x()],
                Point::from_packed_parts(self.inner).y() + 1
            ))
        }
    }
}

/// Iterator that traverse the most likely path down a search tree
pub struct GreedyPath<'a> {
    current: &'a Node,
    threshold: i32
}

impl<'a> GreedyPath<'a> {
    pub fn new(node: &'a Node, threshold: i32) -> GreedyPath<'a> {
        debug_assert!(threshold >= 1);

        GreedyPath { current: node, threshold: threshold }
    }
}

impl<'a> Iterator for GreedyPath<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let max_i = self.current.children.nonzero()
            .max_by(|&a, &b| compare_children(self.current, a, b, MIN_LCB_VISITS))
            .unwrap_or(361);

        if self.current.with(max_i, |child| child.count()) < self.threshold {
            None
        } else {
            unsafe {
                self.current = &*self.current.with(max_i, |child| child.ptr());
            }

            Some(max_i)
        }
    }
}

/// Type alias for `Node` that acts as a wrapper for calling `as_sgf` from
/// within a `write!` macro.
pub struct ToPretty<'a> {
    root: &'a Node,
    verbose: bool
}

impl<'a> fmt::Display for ToPretty<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut children = self.root.children.nonzero().collect::<Vec<usize>>();
        children.sort_by(|&a, &b| compare_children(self.root, b, a, MIN_LCB_VISITS));

        if !self.verbose {
            children.truncate(10);
        }

        // print a summary containing the total tree size
        let total_value: f32 = (0..362)
            .map(|i| self.root.with(i, |child| child.count() as f32 * child.value()))
            .filter(|v| v.is_finite())
            .sum();
        let norm_value = total_value / (self.root.total_count as f32 + 1e-5);
        let likely_path: String = GreedyPath::new(self.root, 1)
                .map(|i| PrettyVertex { inner: i })
                .map(|v| format!("{}", v))
                .collect::<Vec<String>>().join(" ");

        writeln!(fmt, "Nodes: {}, Win: {:.1}%, PV: {}",
            self.root.total_count,
            100.0 * norm_value,
            likely_path
        )?;

        // print a summary of each move that we considered
        for i in children {
            let pretty_vertex = PrettyVertex { inner: i };
            let child = unsafe { &*self.root.with(i, |child| child.ptr()) };
            let likely_path: String = GreedyPath::new(child, 1)
                    .map(|i| PrettyVertex { inner: i })
                    .map(|v| format!("{}", v))
                    .collect::<Vec<String>>().join(" ");

            writeln!(fmt, "{: >5} -> {:7} (W: {:5.2}% / {:5.2}%) (N: {:5.2}%) PV: {} {}",
                     pretty_vertex,
                     child.total_count,
                     100.0 * self.root.with(i, |child| child.value()),
                     100.0 * self.root.with(i, |child| normal_lcb_m(child.value(), child.value_std(), child.count(), self.root.total_count)),
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
pub fn to_pretty(root: &Node) -> ToPretty {
    let verbose = *config::VERBOSE;

    ToPretty { root, verbose }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use dg_go::*;
    use asm::sum_finite_f32;
    use asm::normalize_finite_f32;
    use super::*;

    fn get_prior_distribution(rng: &mut SmallRng, board: &Board, to_move: Color) -> Vec<f32> {
        let mut prior: Vec<f32> = (0..368).map(|_| rng.gen::<f32>()).collect();

        for point in Point::all() {
            if !board.is_valid(to_move, point) {
                prior[point.to_packed_index()] = ::std::f32::NEG_INFINITY;
            }
        }

        let prior_sum = sum_finite_f32(&prior);
        normalize_finite_f32(&mut prior, prior_sum);

        prior
    }

    unsafe fn unsafe_visit_order() {
        let mut choices = vec![];
        let mut rng = SmallRng::from_seed([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mut root = Node::new(
            Color::Black,
            0.5,
            get_prior_distribution(&mut rng, &Board::new(DEFAULT_KOMI), Color::Black)
        );

        loop {
            let trace = probe(&mut root, &mut Board::new(DEFAULT_KOMI));

            if let ProbeResult::Found(trace) = trace {
                assert_eq!(trace.len(), 1);

                // check that we are not re-visiting a node that we have not yet finished
                // expanding.
                let i = trace[0].2;

                assert!(!choices.contains(&i));
                choices.push(i);

                // check that the virtual loss has been correctly applied.
                assert_eq!(root.with(i, |child| child.vcount()), *config::VLOSS_CNT);
                assert_eq!(root.vtotal_count, choices.len() as i32 * *config::VLOSS_CNT as i32);

                // check that all nodes that were visited before this had larger prior
                // value.
                let prior_i = root.prior[i];

                for &other_i in &choices {
                    assert!(root.prior[other_i] >= prior_i);
                }
            } else {
                // check that we did not double-add any virtual loss
                for &other_i in &choices {
                    assert_eq!(root.with(other_i, |child| child.vcount()), *config::VLOSS_CNT);
                }

                assert_eq!(root.vtotal_count, choices.len() as i32 * *config::VLOSS_CNT as i32);
                break;
            }

            assert!(choices.len() <= 362);
        }
    }

    #[test]
    fn visit_order() {
        unsafe { unsafe_visit_order() }
    }

    unsafe fn unsafe_virtual_loss() {
        let mut board = Board::new(DEFAULT_KOMI);
        let mut rng = SmallRng::from_seed([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mut root = Node::new(
            Color::Black,
            0.5,
            get_prior_distribution(&mut rng, &board, Color::Black)
        );

        if let ProbeResult::Found(trace) = probe(&mut root, &mut board) {
            let i = trace[0].2;

            // check that the virtual loss was applied
            assert_eq!(root.with(i, |child| child.vcount()), *config::VLOSS_CNT);
            assert_eq!(root.vtotal_count, *config::VLOSS_CNT as i32);

            // check that the virtual loss is un-applied after we update this move, and
            // that we we increase the `count` instead.
            let other_prior = get_prior_distribution(&mut rng, &board, Color::Black);
            let other_value = 0.9;

            insert(&trace, Color::Black, other_value, other_prior);

            assert_eq!(root.with(i, |child| child.vcount()), 0);
            assert_eq!(root.vtotal_count, 0);
            assert_eq!(root.with(i, |child| child.count()), 1);
            assert_eq!(root.total_count, 1);
        } else {
            panic!();
        }
    }

    #[test]
    fn virtual_loss() {
        unsafe { unsafe_virtual_loss() }
    }

    unsafe fn unsafe_value_update() {
        let mut board = Board::new(DEFAULT_KOMI);
        let mut root = Node::new(
            Color::Black,
            0.5,
            (0..362).map(|i| if i == 60 { 1.0 } else { 0.0 }).collect()
        );

        // to setup a scenario where we have two parallel probes that will both update
        // the same node value we need to pre-expand a node.
        let other_prior: Vec<f32> = (0..362).map(|i| if i == 61 || i == 62 { 0.5 } else { 0.0 }).collect();
        let trace = probe(&mut root, &mut board).unwrap();

        insert(&trace, Color::Black, 0.9, other_prior.clone());
        assert!({
            let value = root.with(60, |child| child.value());

            value >= 0.8999 && value <= 0.9001
        });
        assert_eq!(root.with(60, |child| child.count()), 1);
        assert_eq!(root.total_count, 1);
        assert_eq!(root.with(60, |child| child.vcount()), 0);
        assert_eq!(root.vtotal_count, 0);

        // two parallel probes in the same sub-tree.
        let trace_1 = probe(&mut root, &mut Board::new(DEFAULT_KOMI)).unwrap();
        let trace_2 = probe(&mut root, &mut Board::new(DEFAULT_KOMI)).unwrap();

        assert_eq!(trace_1[0].2, 60);
        assert_eq!(trace_2[0].2, 60);
        assert!(trace_1[1].2 == 61 || trace_2[1].2 == 61);
        assert!(trace_1[1].2 == 62 || trace_2[1].2 == 62);

        // the value of the root sub-tree should remain unchanged, but the virtual loss
        // should have increased.
        assert_eq!(root.with(60, |child| child.value()), 0.9);
        assert_eq!(root.with(60, |child| child.count()), 1);
        assert_eq!(root.total_count, 1);
        assert_eq!(root.with(60, |child| child.vcount()), 2 * *config::VLOSS_CNT);
        assert_eq!(root.vtotal_count, 2 * *config::VLOSS_CNT as i32);

        // check update after the first probe is inserted
        insert(&trace_1, Color::White, 0.2, other_prior.clone());

        assert_eq!(root.with(60, |child| child.value()), 0.85);
        assert_eq!(root.with(60, |child| child.count()), 2);
        assert_eq!(root.total_count, 2);
        assert_eq!(root.with(60, |child| child.vcount()), *config::VLOSS_CNT);
        assert_eq!(root.vtotal_count, *config::VLOSS_CNT as i32);

        // check update after the second probe is inserted
        insert(&trace_2, Color::White, 0.3, other_prior.clone());

        assert_eq!(root.with(60, |child| child.value()), 0.8);
        assert_eq!(root.with(60, |child| child.count()), 3);
        assert_eq!(root.total_count, 3);
        assert_eq!(root.with(60, |child| child.vcount()), 0);
        assert_eq!(root.vtotal_count, 0);
    }

    #[test]
    fn value_update() {
        unsafe { unsafe_value_update() }
    }

    unsafe fn unsafe_undo_trace() {
        let mut board = Board::new(DEFAULT_KOMI);
        let mut root = Node::new(
            Color::Black,
            0.5,
            (0..362).map(|i| if i == 60 { 1.0 } else { 0.0 }).collect()
        );

        // probe twice, of which the first will be undone, then check that the tree is
        // consistent with this.
        assert!(probe(&mut root, &mut board).is_some());
        assert!(probe(&mut root, &mut board).is_none());

        assert_eq!(root.vtotal_count, *config::VLOSS_CNT as i32);
    }

    #[test]
    fn undo_trace() {
        unsafe { unsafe_undo_trace() }
    }
}
