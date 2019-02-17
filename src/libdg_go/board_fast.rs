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

use color::Color;
use zobrist;

use std::marker::PhantomData;

pub trait Counter {
    type Output;

    fn get(&self) -> Self::Output;

    /// Adds the given liberty to this counter, and if this liberty was not
    /// already part of this counter returns true.
    /// 
    /// # Arguments
    /// 
    /// * `index`
    /// 
    fn add(&mut self, index: usize) -> bool;
}

/// A counter that is capable of counting up to one.
pub struct One {
    other: usize
}

impl Default for One {
    fn default() -> One {
        One { other: 0xffff }
    }
}

impl Counter for One {
    type Output = usize;

    fn get(&self) -> Self::Output { self.other }
    fn add(&mut self, index: usize) -> bool {
        self.other = index;
        true
    }
}

/// A counter that is capable of counting up to two.
pub struct Two {
    other: usize
}

impl Default for Two {
    fn default() -> Two {
        Two { other: 0xffff }
    }
}

impl Counter for Two {
    type Output = usize;

    fn get(&self) -> Self::Output { self.other }
    fn add(&mut self, index: usize) -> bool {
        if index != self.other {
            self.other = index;
            true
        } else {
            false
        }
    }
}

/// A counter that is capable of counting up to three.
pub struct Three {
    other: [u16; 3],
    count: usize
}

impl Default for Three {
    fn default() -> Three {
        Three {
            other: [0xffff; 3],
            count: 0
        }
    }
}

impl Counter for Three {
    type Output = [u16; 3];

    fn get(&self) -> Self::Output { self.other }
    fn add(&mut self, index: usize) -> bool {
        if !self.other.contains(&(index as u16)) {
            self.other[self.count] = index as u16;
            self.count += 1;

            true
        } else {
            false
        }
    }
}

pub struct N {
    liberties: Vec<usize>
}

impl Default for N {
    fn default() -> N {
        N { liberties: vec! [] }
    }
}

impl Counter for N {
    type Output = Vec<usize>;

    fn get(&self) -> Self::Output { self.liberties.clone() }
    fn add(&mut self, index: usize) -> bool {
        if !self.liberties.contains(&index) {
            self.liberties.push(index);

            true
        } else {
            false
        }
    }
}

/// Iterator over all vertices that are directly adjacent to the given one. It will
/// always return four values:
///
/// - North
/// - East
/// - South
/// - West
///
pub struct AdjacentIter(usize, usize);
pub struct AdjacentVertexIter<'a>(AdjacentIter, &'a BoardFast);

impl AdjacentIter {
    pub fn new(source: usize) -> AdjacentIter {
        AdjacentIter(source, 0)
    }

    pub fn with_vertex(self, board: &BoardFast) -> AdjacentVertexIter {
        AdjacentVertexIter(self, board)
    }
}

impl Iterator for AdjacentIter {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.1;

        if index < 4 {
            self.1 += 1;

            Some(::codegen::NESW[self.0][index] as usize)
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (4, Some(4))
    }
}

impl<'a> Iterator for AdjacentVertexIter<'a> {
    type Item = (usize, u16);

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            None => None,
            Some(index) => {
                let vertex = unsafe { *self.1.vertices.get_unchecked(index) };

                Some((index, vertex))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

/// Iterator over all vertices of the same color that are strongly connected to the
/// given starting point.
pub struct BlockIter<'a> {
    board: &'a BoardFast,
    starting_point: usize,
    current_point: Option<usize>
}

impl<'a> BlockIter<'a> {
    fn new(board: &'a BoardFast, starting_point: usize) -> BlockIter<'a> {
        let current_point = Some(starting_point);

        BlockIter { board, starting_point, current_point }
    }
}

impl<'a> Iterator for BlockIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current_point) = self.current_point {
            let next_point = unsafe {
                self.board.vertices.get_unchecked(current_point).next_vertex() as usize
            };

            self.current_point = if next_point != self.starting_point {
                Some(next_point)
            } else {
                None
            };

            Some(current_point)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(361))
    }

    fn any<F: FnMut(Self::Item) -> bool>(&mut self, mut f: F) -> bool {
        if let Some(mut current_point) = self.current_point {
            loop {
                if f(current_point) {
                    return true;
                }

                current_point = unsafe {
                    self.board.vertices.get_unchecked(current_point).next_vertex() as usize
                };

                if current_point == self.starting_point {
                    return false;
                }
            }
        } else {
            false
        }
    }
}

/// Iterator over all vertices, with a mutable reference, of the same color that are strongly
/// connected to the given starting point.
pub struct BlockIterMut<'a> {
    board: *mut BoardFast,
    starting_point: usize,
    current_point: Option<usize>,
    phantom_ref: PhantomData<&'a mut BoardFast>
}

impl<'a> BlockIterMut<'a> {
    fn new(board: &'a mut BoardFast, starting_point: usize) -> BlockIterMut<'a> {
        let current_point = Some(starting_point);
        let phantom_ref = PhantomData::default();

        BlockIterMut { board, starting_point, current_point, phantom_ref }
    }
}

impl<'a> Iterator for BlockIterMut<'a> {
    type Item = (usize, &'a mut u16);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current_point) = self.current_point {
            let vertex = unsafe {
                (*self.board).vertices.get_unchecked_mut(current_point)
            };
            let next_point = vertex.next_vertex() as usize;

            self.current_point = if next_point != self.starting_point {
                Some(next_point)
            } else {
                None
            };

            Some((current_point, vertex))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(361))
    }
}

/// Representation of a set of strongly connected vertices of the same color.
pub struct Block<'a> {
    board: &'a BoardFast,
    starting_point: usize
}

impl<'a> Block<'a> {
    fn new(board: &'a BoardFast, starting_point: usize) -> Block<'a> {
        Block { board, starting_point }
    }
}

impl<'a> IntoIterator for Block<'a> {
    type Item = <BlockIter<'a> as Iterator>::Item;
    type IntoIter = BlockIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        BlockIter::new(self.board, self.starting_point)
    }
}

pub struct BlockMut<'a> {
    board: &'a mut BoardFast,
    starting_point: usize
}

impl<'a> BlockMut<'a> {
    fn new(board: &'a mut BoardFast, starting_point: usize) -> BlockMut<'a> {
        BlockMut { board, starting_point }
    }
}

impl<'a> IntoIterator for BlockMut<'a> {
    type Item = <BlockIterMut<'a> as Iterator>::Item;
    type IntoIter = BlockIterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        BlockIterMut::new(self.board, self.starting_point)
    }
}

pub trait Vertex {
    fn color(self) -> u8;
    fn set_color(&mut self, color: u8);
    fn next_vertex(self) -> u16;
    fn set_next_vertex(&mut self, next_vertex: u16);
    fn visited(self) -> bool;
    fn set_visited(&mut self, visited: bool);
}

impl Vertex for u16 {
    #[inline(always)]
    fn color(self) -> u8 {
        (self & 0x3) as u8
    }

    #[inline(always)]
    fn set_color(&mut self, color: u8) {
        debug_assert!(color <= 3);

        *self = *self & 0xfffc | color as u16;
    }

    #[inline(always)]
    fn next_vertex(self) -> u16 {
        (self & 0x7fff) >> 2
    }

    #[inline(always)]
    fn set_next_vertex(&mut self, next_vertex: u16) {
        *self = (*self & 0x8003) | (next_vertex << 2);
    }

    #[inline(always)]
    fn visited(self) -> bool {
        (self & 0x8000) != 0
    }

    #[inline(always)]
    fn set_visited(&mut self, visited: bool) {
        *self = (*self & 0x7fff) | if visited { 0x8000 } else { 0 }
    }
}

/// Minimal representation of a go board that implements all rules (except super-ko).
#[derive(Clone)]
pub struct BoardFast {
    /// Packed bit structure that contains the following fields. It has been padded
    /// with additional elements at the end that are used instead of out-of-bounds
    /// checks.
    ///
    /// - `color` - 2 bits
    /// - `next_vertex` - 13 bits
    /// - `visited` - 1 bit
    ///
    pub vertices: [u16; 368],
}

impl BoardFast {
    /// Returns an empty board.
    pub fn new() -> BoardFast {
        let mut board = BoardFast {
            vertices: [0; 368],
        };

        // fill the padding with _invalid_ elements that does not match either
        // of the three possible vertices (`Black`, `White`, and `Empty`).
        for i in 361..368 {
            board.vertices[i].set_color(0x3);
        }

        board
    }

    /// Returns an iterator over all vertices that are adjacent to the given
    /// vertex.
    ///
    /// # Arguments
    ///
    /// * `at_index` -
    ///
    pub fn adjacent_to(&self, at_index: usize) -> AdjacentVertexIter {
        AdjacentIter::new(at_index).with_vertex(self)
    }

    /// Returns the block, set of strongly connected vertices of the same
    /// color, at the given vertex.
    ///
    /// # Arguments
    ///
    /// * `at_index` -
    ///
    pub fn block_at(&self, at_index: usize) -> Block {
        Block::new(self, at_index)
    }

    /// Returns the block, set of strongly connected vertices of the same
    /// color, at the given vertex with a mutable reference to each vertex.
    ///
    /// # Arguments
    ///
    /// * `at_index` -
    ///
    pub fn block_at_mut(&mut self, at_index: usize) -> BlockMut {
        BlockMut::new(self, at_index)
    }

    /// Returns whether the given liberties of the given group (as counted by
    /// the given counter). It will stop counting after `n` liberties has been
    /// found.
    /// 
    /// # Arguments
    /// 
    /// * `index` - the index of a vertex in the group
    /// * `n` - the maximum number of liberties to count
    /// 
    #[inline]
    pub fn get_n_liberty<C: Counter + Default>(
        &self,
        index: usize,
        mut n: usize
    ) -> C::Output
    {
        let mut counter = C::default();

        for current in self.block_at(index) {
            for (other_index, other_vertex) in self.adjacent_to(current) {
                if other_vertex.color() == 0 && counter.add(other_index) {
                    n -= 1;

                    if n == 0 {
                        return counter.get();
                    }
                }
            }
        }

        counter.get()
    }

    /// Returns whether the given group has at least `n` liberties, using the
    /// given counter to do so.
    /// 
    /// # Arguments
    /// 
    /// * `index` - the index of a vertex in the group
    /// * `n` - the maximum number of liberties to count
    /// 
    #[inline]
    pub fn has_n_liberty<C: Counter + Default>(&self, index: usize, mut n: usize) -> bool {
        let mut counter = C::default();

        for current in self.block_at(index) {
            for (other_index, other_vertex) in self.adjacent_to(current) {
                if other_vertex.color() == 0 && counter.add(other_index) {
                    n -= 1;

                    if n == 0 {
                        return true
                    }
                }
            }
        }

        false
    }

    /// Returns whether the given group has at least `n` liberties, using the
    /// given counter to do so.
    ///
    /// # Arguments
    ///
    /// * `index` - the index of a vertex in the group
    /// * `n` - the maximum number of liberties to count
    /// * `workspace` - the memoization of the board liberties
    ///
    #[inline]
    pub fn has_n_liberty_mut<C: Counter + Default>(&self, index: usize, mut n: usize, workspace: &mut [u8]) -> bool {
        if workspace[index] != 0 {
            workspace[index] == 0xff
        } else {
            let mut counter = C::default();

            for current in self.block_at(index) {
                workspace[current] = 1;

                for (other_index, other_vertex) in self.adjacent_to(current) {
                    if other_vertex.color() == 0 && counter.add(other_index) {
                        n -= 1;

                        if n == 0 {
                            self.set_workspace_mut(index, 0xff, workspace);
                            return true
                        }
                    }
                }
            }

            false
        }
    }

    /// Update the memoization value for all vertices connected to `index` to `value`.
    ///
    /// # Arguments
    ///
    /// * `index` -
    /// * `value` -
    /// * `workspace` -
    ///
    fn set_workspace_mut(&self, index: usize, value: u8, workspace: &mut [u8]) {
        for current in self.block_at(index) {
            workspace[current] = value;
        }
    }

    /// Returns whether the given move is valid according to the
    /// Tromp-Taylor rules.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `index` - the HW index of the move
    ///
    pub fn is_valid(&self, color: Color, index: usize) -> bool {
        self.vertices[index].color() == 0 && {
            let current = color as u8;

            for (other_index, other_vertex) in self.adjacent_to(index) {
                let value = other_vertex.color();

                // check for direct liberties
                if value == 0 {
                    return true;
                }

                // check for the following two conditions simplified into one case:
                //
                // 1. If a neighbour is friendly then we are fine if it has at
                //    least two liberties.
                // 2. If a neighbour is unfriendly then we are fine if it has less
                //    than two liberties (i.e. one).
                if value != 0x3 && (value == current) == self.has_n_liberty::<Two>(other_index, 2) {
                    return true;
                }
            }

            false  // move is suicide :'(
        }
    }

    /// Returns whether the given move is valid according to the
    /// Tromp-Taylor rules.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `index` - the HW index of the move
    /// * `workspace` - the memoization of the board liberties
    ///
    #[inline(always)]
    pub fn is_valid_mut(&self, color: Color, index: usize, workspace: &mut [u8]) -> bool {
        self.vertices[index].color() == 0 && {
            let current = color as u8;

            for (other_index, other_vertex) in self.adjacent_to(index) {
                let value = other_vertex.color();

                // check for direct liberties
                if value == 0 {
                    return true;
                }

                // check for the following two conditions simplified into one case:
                //
                // 1. If a neighbour is friendly then we are fine if it has at
                //    least two liberties.
                // 2. If a neighbour is unfriendly then we are fine if it has less
                //    than two liberties (i.e. one).
                if value != 0x3 && (value == current) == self.has_n_liberty_mut::<Two>(other_index, 2, workspace) {
                    return true;
                }
            }

            false  // move is suicide :'(
        }
    }

    /// Connects the chains of the two vertices into one chain. This method
    /// should not be called with the same group twice as that will result
    /// in a corrupted chain.
    ///
    /// # Arguments
    ///
    /// * `next_vertex` - the array containing the next vertices
    /// * `index` - the first chain to connect
    /// * `other` - the second chain to connect
    ///
    #[inline]
    fn join_vertices(&mut self, index: usize, other: usize) {
        // check so that other is not already in the chain starting
        // at index since that would lead to a corrupted chain.
        if self.block_at(index).into_iter().any(|current| current == other) {
            return
        }

        // re-connect the two lists so if we have two chains `A` and `B`:
        //
        //   A:  a -> b -> c -> a
        //   B:  1 -> 2 -> 3 -> 1
        //
        // then the final new chain will be:
        //
        //   a -> 2 -> 3 -> 1 -> b -> c -> a
        //
        let index_prev = self.vertices[index].next_vertex();
        let other_prev = self.vertices[other].next_vertex();

        self.vertices[other].set_next_vertex(index_prev);
        self.vertices[index].set_next_vertex(other_prev);
    }

    /// Returns the zobrist hash adjustment that would need to be done if the
    /// group at the given index was capture and was of the given color.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the group to capture
    /// * `index` - the index of a stone in the group
    ///
    #[inline]
    pub fn capture_if(&self, color: usize, index: usize) -> u64 {
        let mut adjust = 0;

        for current in self.block_at(index) {
            adjust ^= zobrist::TABLE[color][current];
        }

        adjust
    }

    /// Remove all stones strongly connected to the given index from the
    /// board. It returns the necessary adjustment to the zobrist hash.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the group to capture
    /// * `index` - the index of a stone in the group to capture
    ///
    #[inline]
    pub fn capture(&mut self, color: usize, index: usize) -> u64 {
        let mut hash = 0;

        for (other_index, other_vertex) in self.block_at_mut(index) {
            hash ^= zobrist::TABLE[color][other_index];
            other_vertex.set_color(0);
        }

        hash
    }

    /// Returns the zobrist hash adjustments that are would be made if a stone
    /// of the given color was played on the given vertex.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `index` - the HW index of the move
    ///
    #[inline]
    pub fn place_if(&self, color: Color, index: usize) -> u64 {
        let opponent = color.opposite() as u8;
        let mut adjust = zobrist::TABLE[color as usize][index];

        for (other_index, other_vertex) in self.adjacent_to(index) {
            if other_vertex.color() == opponent && !self.has_n_liberty::<Two>(other_index, 2) {
                adjust ^= self.capture_if(opponent as usize, other_index);
            }
        }

        adjust
    }

    /// Returns the zobrist hash adjustments that are would be made if a stone
    /// of the given color was played on the given vertex.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `index` - the HW index of the move
    /// * `workspace` - the memoization of the board liberties
    ///
    #[inline]
    pub fn place_if_mut(&self, color: Color, index: usize, workspace: &mut [u8]) -> u64 {
        let opponent = color.opposite() as u8;
        let mut adjust = zobrist::TABLE[color as usize][index];

        for (other_index, other_vertex) in self.adjacent_to(index) {
            if other_vertex.color() == opponent && !self.has_n_liberty_mut::<Two>(other_index, 2, workspace) {
                adjust ^= self.capture_if(opponent as usize, other_index);
            }
        }

        adjust
    }

    /// Place a some of the given `color` at the given `index` on this board. This function
    /// assume that the given move is valid.
    ///
    /// # Arguments
    ///
    /// * `color` -
    /// * `index` -
    ///
    #[inline]
    pub fn place(&mut self, color: Color, index: usize) -> u64 {
        let player = color as u8;

        // place the stone on the board regardless of whether it is legal
        // or not.
        self.vertices[index].set_color(color as u8);
        self.vertices[index].set_next_vertex(index as u16);
        self.vertices[index].set_visited(true);

        // connect this stone to any neighbouring groups
        for other_index in AdjacentIter::new(index) {
            let value = unsafe { self.vertices.get_unchecked(other_index).color() };

            if value == player {
                self.join_vertices(index, other_index);
            }
        }

        // clear the opponents color
        let opponent = color.opposite() as u8;
        let mut hash = zobrist::TABLE[color as usize][index];

        for other_index in AdjacentIter::new(index) {
            let value = unsafe { self.vertices.get_unchecked(other_index).color() };

            if value == opponent && !self.has_n_liberty::<One>(other_index, 1) {
                hash ^= self.capture(opponent as usize, other_index);
            }
        }

        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_vertex_bitfield() {
        let mut x: u16 = 0;

        for color in 0..4 {
            x.set_color(color);

            for next_vertex in 0..362 {
                x.set_next_vertex(next_vertex);

                for visited in vec! [true, false] {
                    x.set_visited(visited);

                    assert_eq!(x.color(), color);
                    assert_eq!(x.next_vertex(), next_vertex);
                    assert_eq!(x.visited(), visited);
                }
            }
        }
    }
}
