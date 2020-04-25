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
use point::Point;
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
    fn add(&mut self, point: Point) -> bool;
}

/// A counter that is capable of counting up to one.
pub struct One {
    other: Point
}

impl Default for One {
    fn default() -> One {
        One { other: Point::default() }
    }
}

impl Counter for One {
    type Output = Point;

    fn get(&self) -> Self::Output { self.other }
    fn add(&mut self, point: Point) -> bool {
        self.other = point;
        true
    }
}

/// A counter that is capable of counting up to two.
pub struct Two {
    other: Point
}

impl Default for Two {
    fn default() -> Two {
        Two { other: Point::default() }
    }
}

impl Counter for Two {
    type Output = Point;

    fn get(&self) -> Self::Output { self.other }
    fn add(&mut self, point: Point) -> bool {
        if point != self.other {
            self.other = point;
            true
        } else {
            false
        }
    }
}

/// A counter that is capable of counting up to three.
pub struct Three {
    other: [Point; 3],
    count: usize
}

impl Default for Three {
    fn default() -> Three {
        Three {
            other: [Point::default(); 3],
            count: 0
        }
    }
}

impl Counter for Three {
    type Output = [Point; 3];

    fn get(&self) -> Self::Output { self.other }
    fn add(&mut self, point: Point) -> bool {
        if !self.other.contains(&point) {
            self.other[self.count] = point;
            self.count += 1;

            true
        } else {
            false
        }
    }
}

pub struct N {
    liberties: Vec<Point>
}

impl Default for N {
    fn default() -> N {
        N { liberties: vec! [] }
    }
}

impl Counter for N {
    type Output = Vec<Point>;

    fn get(&self) -> Self::Output { self.liberties.clone() }
    fn add(&mut self, point: Point) -> bool {
        if !self.liberties.contains(&point) {
            self.liberties.push(point);

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
pub struct AdjacentIter(Point, usize);
pub struct AdjacentVertexIter<'a>(AdjacentIter, &'a BoardFast);

impl AdjacentIter {
    pub fn new(source: Point) -> AdjacentIter {
        AdjacentIter(source, 0)
    }

    pub fn with_vertex(self, board: &BoardFast) -> AdjacentVertexIter {
        AdjacentVertexIter(self, board)
    }
}

impl Iterator for AdjacentIter {
    type Item = Point;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.1;

        if index < 4 {
            const DX: [isize; 4] = [1, -1, 0,  0];
            const DY: [isize; 4] = [0,  0, 1, -1];

            self.1 += 1;
            Some(self.0.offset(DX[index], DY[index]))
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
    type Item = (Point, u16);

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            None => None,
            Some(point) => {
                let vertex = self.1.vertices[point];

                Some((point, vertex))
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
    starting_point: Point,
    current_point: Option<Point>
}

impl<'a> BlockIter<'a> {
    fn new(board: &'a BoardFast, starting_point: Point) -> BlockIter<'a> {
        let current_point = Some(starting_point);

        BlockIter { board, starting_point, current_point }
    }
}

impl<'a> Iterator for BlockIter<'a> {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current_point) = self.current_point {
            let next_point = self.board.vertices[current_point].next_vertex();

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

                current_point = self.board.vertices[current_point].next_vertex();

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
    starting_point: Point,
    current_point: Option<Point>,
    phantom_ref: PhantomData<&'a mut BoardFast>
}

impl<'a> BlockIterMut<'a> {
    fn new(board: &'a mut BoardFast, starting_point: Point) -> BlockIterMut<'a> {
        let current_point = Some(starting_point);
        let phantom_ref = PhantomData::default();

        BlockIterMut { board, starting_point, current_point, phantom_ref }
    }
}

impl<'a> Iterator for BlockIterMut<'a> {
    type Item = (Point, &'a mut u16);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current_point) = self.current_point {
            let vertex = unsafe { &mut (*self.board).vertices[current_point] };
            let next_point = vertex.next_vertex();

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
    starting_point: Point
}

impl<'a> Block<'a> {
    fn new(board: &'a BoardFast, starting_point: Point) -> Block<'a> {
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
    starting_point: Point
}

impl<'a> BlockMut<'a> {
    fn new(board: &'a mut BoardFast, starting_point: Point) -> BlockMut<'a> {
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
    fn next_vertex(self) -> Point;
    fn set_next_vertex(&mut self, next_vertex: Point);
    fn visited(self) -> bool;
    fn set_visited(&mut self, visited: bool);
}

const EMPTY: u16 = 0x0;
const INVALID: u16 = 0x3;

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
    fn next_vertex(self) -> Point {
        Point::from_raw_parts((self & 0x7fff) >> 2)
    }

    #[inline(always)]
    fn set_next_vertex(&mut self, next_vertex: Point) {
        *self = (*self & 0x8003) | ((next_vertex.to_i() as u16) << 2);
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
    pub vertices: [u16; Point::MAX],
}

impl BoardFast {
    /// Returns an empty board.
    pub fn new() -> BoardFast {
        let mut board = BoardFast {
            vertices: [INVALID; Point::MAX],
        };

        for point in Point::all() {
            board.vertices[point] = EMPTY;
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
    pub fn adjacent_to(&self, at_point: Point) -> AdjacentVertexIter {
        AdjacentIter::new(at_point).with_vertex(self)
    }

    /// Returns the block, set of strongly connected vertices of the same
    /// color, at the given vertex.
    ///
    /// # Arguments
    ///
    /// * `at_point` -
    ///
    pub fn block_at(&self, at_point: Point) -> Block {
        Block::new(self, at_point)
    }

    /// Returns the block, set of strongly connected vertices of the same
    /// color, at the given vertex with a mutable reference to each vertex.
    ///
    /// # Arguments
    ///
    /// * `at_point` -
    ///
    pub fn block_at_mut(&mut self, at_point: Point) -> BlockMut {
        BlockMut::new(self, at_point)
    }

    /// Returns whether the given liberties of the given group (as counted by
    /// the given counter). It will stop counting after `n` liberties has been
    /// found.
    /// 
    /// # Arguments
    /// 
    /// * `at_point` - the index of a vertex in the group
    /// * `n` - the maximum number of liberties to count
    /// 
    #[inline]
    pub fn get_n_liberty<C: Counter + Default>(
        &self,
        at_point: Point,
        mut n: usize
    ) -> C::Output
    {
        let mut counter = C::default();

        for current in self.block_at(at_point) {
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
    /// * `at_point` - the index of a vertex in the group
    /// * `n` - the maximum number of liberties to count
    /// 
    #[inline]
    pub fn has_n_liberty<C: Counter + Default>(&self, at_point: Point, mut n: usize) -> bool {
        let mut counter = C::default();

        for current in self.block_at(at_point) {
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
    /// * `at_point` - the index of a vertex in the group
    /// * `n` - the maximum number of liberties to count
    /// * `workspace` - the memoization of the board liberties
    ///
    #[inline]
    pub fn has_n_liberty_mut<C: Counter + Default>(&self, at_point: Point, mut n: usize, workspace: &mut [u8]) -> bool {
        if workspace[at_point] != 0 {
            workspace[at_point] == 0xff
        } else {
            let mut counter = C::default();

            for current in self.block_at(at_point) {
                workspace[current] = 1;

                for (other_index, other_vertex) in self.adjacent_to(current) {
                    if other_vertex.color() == 0 && counter.add(other_index) {
                        n -= 1;

                        if n == 0 {
                            self.set_workspace_mut(at_point, 0xff, workspace);
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
    fn set_workspace_mut(&self, at_point: Point, value: u8, workspace: &mut [u8]) {
        for current in self.block_at(at_point) {
            workspace[current] = value;
        }
    }

    /// Returns whether the given move is valid according to the
    /// Tromp-Taylor rules.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `at_point` - the HW index of the move
    ///
    pub fn is_valid(&self, color: Color, at_point: Point) -> bool {
        self.vertices[at_point].color() == 0 && {
            let current = color as u8;

            for (other_index, other_vertex) in self.adjacent_to(at_point) {
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
    /// * `at_point` - the HW index of the move
    /// * `workspace` - the memoization of the board liberties
    ///
    #[inline(always)]
    pub fn is_valid_mut(&self, color: Color, at_point: Point, workspace: &mut [u8]) -> bool {
        self.vertices[at_point].color() == 0 && {
            let current = color as u8;

            for (other_index, other_vertex) in self.adjacent_to(at_point) {
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
    /// * `one` - the first chain to connect
    /// * `two` - the second chain to connect
    ///
    #[inline]
    fn join_vertices(&mut self, one: Point, two: Point) {
        // check so that other is not already in the chain starting
        // at index since that would lead to a corrupted chain.
        if self.block_at(one).into_iter().any(|current| current == two) {
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
        let one_prev = self.vertices[one].next_vertex();
        let two_prev = self.vertices[two].next_vertex();

        self.vertices[two].set_next_vertex(one_prev);
        self.vertices[one].set_next_vertex(two_prev);
    }

    /// Returns the zobrist hash adjustment that would need to be done if the
    /// group at the given index was capture and was of the given color.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the group to capture
    /// * `at_point` - the index of a stone in the group
    ///
    #[inline]
    pub fn capture_if(&self, color: usize, at_point: Point) -> u64 {
        let mut adjust = 0;

        for current in self.block_at(at_point) {
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
    /// * `at_point` - the index of a stone in the group to capture
    ///
    #[inline]
    pub fn capture(&mut self, color: usize, at_point: Point) -> u64 {
        let mut hash = 0;

        for (other_index, other_vertex) in self.block_at_mut(at_point) {
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
    /// * `at_point` - the HW index of the move
    ///
    #[inline]
    pub fn place_if(&self, color: Color, at_point: Point) -> u64 {
        let opponent = color.opposite() as u8;
        let mut adjust = zobrist::TABLE[color as usize][at_point];

        for (other_index, other_vertex) in self.adjacent_to(at_point) {
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
    /// * `at_point` - the HW index of the move
    /// * `workspace` - the memoization of the board liberties
    ///
    #[inline]
    pub fn place_if_mut(&self, color: Color, at_point: Point, workspace: &mut [u8]) -> u64 {
        let opponent = color.opposite() as u8;
        let mut adjust = zobrist::TABLE[color as usize][at_point];

        for (other_index, other_vertex) in self.adjacent_to(at_point) {
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
    /// * `at_point` -
    ///
    #[inline]
    pub fn place(&mut self, color: Color, at_point: Point) -> u64 {
        let player = color as u8;

        // place the stone on the board regardless of whether it is legal
        // or not.
        self.vertices[at_point].set_color(color as u8);
        self.vertices[at_point].set_next_vertex(at_point);
        self.vertices[at_point].set_visited(true);

        // connect this stone to any neighbouring groups
        for other_point in AdjacentIter::new(at_point) {
            let value = self.vertices[other_point].color();

            if value == player {
                self.join_vertices(at_point, other_point);
            }
        }

        // clear the opponents color
        let opponent = color.opposite() as u8;
        let mut hash = zobrist::TABLE[color as usize][at_point];

        for other_point in AdjacentIter::new(at_point) {
            let value = self.vertices[other_point].color();

            if value == opponent && !self.has_n_liberty::<One>(other_point, 1) {
                hash ^= self.capture(opponent as usize, other_point);
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

            for next_vertex in Point::all() {
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
