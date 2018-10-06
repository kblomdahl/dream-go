// Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

/// Macro for iterating over directions around a point.
/// 
/// ```
/// let index: usize = 100;
/// 
/// foreach_nd!(index, |other_index, value| {
///     // ...
/// }, N, E, S, W)
/// ```
macro_rules! foreach_nd {
    ($this:expr, $source:expr, |$index:ident, $value:ident| $stmt:block, $($dir:ident),*) => ({
        $(
            let $index = unsafe {
                *::codegen::$dir.get_unchecked($source) as usize
            };
            let $value = unsafe {
                *$this.vertices.get_unchecked($index)
            };

            $stmt
        );*
    })
}

macro_rules! foreach_4d {
    ($this:expr, $source:expr, |$index:ident, $value:ident| $stmt:block) => {{
        foreach_nd!($this, $source, |$index, $value| $stmt, N, E, S, W)
    }};
}

/// Macro for finding a matching directions around a point.
/// 
/// ```
/// let index: usize = 100;
/// let other_index = find_nd!(index, |other_index, value| {
///     Some(0)
/// }, N, E, S, W)
/// 
/// assert_eq!(other_index, Some(0));
/// ```
macro_rules! find_nd {
    ($this:expr, $source:expr, |$index:ident, $value:ident| $stmt:block) => ({
        None
    });

    ($this:expr, $source:expr, |$index:ident, $value:ident| $stmt:block, $dir:ident $(,$rest:ident)*) => ({
        let __result = {
            let $index = unsafe {
                *::codegen::$dir.get_unchecked($source) as usize
            };
            let $value = unsafe {
                *$this.vertices.get_unchecked($index)
            };

            $stmt
        };

        if __result.is_some() {
            __result
        } else {
            find_nd!($this, $source, |$index, $value| $stmt $(,$rest)*)
        }
    })
}

macro_rules! find_4d {
    ($this:expr, $source:expr, |$index:ident, $value:ident| $stmt:block) => {{
        find_nd!($this, $source, |$index, $value| $stmt, N, E, S, W)
    }};
}

/// 
#[derive(Clone)]
pub struct BoardFast {
    /// The color of the stone that is occupying each vertex. This array
    /// should in addition contain at least one extra padding element that
    /// contains `0xff`, this extra element is used to the out-of-bounds
    /// index to avoid extra branches.
    pub vertices: [u8; 368],

    /// The index of a stone that is strongly connected to each vertex in
    /// such a way that every stone in a strongly connected group forms
    /// a cycle.
    pub next_vertex: [u16; 361]
}

impl BoardFast {
    pub fn new() -> BoardFast {
        let mut board = BoardFast {
            vertices: [0; 368],
            next_vertex: [0; 361]
        };

        // fill the padding with _invalid_ elements that does not match either
        // of the three possible vertices (`Black`, `White`, and `Empty`).
        for i in 361..368 {
            board.vertices[i] = 0xff;
        }

        board
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
        let mut current = index;

        loop {
            foreach_4d!(self, current, |other_index, value| {
                if value == 0 && counter.add(other_index) {
                    n -= 1;

                    if n == 0 {
                        return counter.get();
                    }
                }
            });

            current = unsafe { *self.next_vertex.get_unchecked(current) as usize };
            if current == index {
                break;
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
        let mut current = index;

        loop {
            foreach_4d!(self, current, |other_index, value| {
                if value == 0 && counter.add(other_index) {
                    n -= 1;

                    if n == 0 {
                        return true
                    }
                }
            });

            current = unsafe { *self.next_vertex.get_unchecked(current) as usize };
            if current == index {
                break;
            }
        }

        false
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
        self.vertices[index] == 0 && {
            let current = color as u8;

            foreach_4d!(self, index, |other_index, value| {
                // check for direct liberties
                if value == 0 {
                    return true;
                }

                // check for the following two conditions simplied into one case:
                //
                // 1. If a neighbour is friendly then we are fine if it has at
                //    least two liberties.
                // 2. If a neighbour is unfriendly then we are fine if it has less
                //    than two liberties (i.e. one).
                if value != 0xff && (value == current) == self.has_n_liberty::<Two>(other_index, 2) {
                    return true;
                }
            });

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
        let mut current = index;

        loop {
            if current == other {
                return;
            }

            current = self.next_vertex[current] as usize;
            if current == index {
                break;
            }
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
        let index_prev = self.next_vertex[index];
        let other_prev = self.next_vertex[other];

        self.next_vertex[other] = index_prev;
        self.next_vertex[index] = other_prev;
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
        let mut current = index;
        let mut adjust = 0;

        loop {
            adjust ^= zobrist::TABLE[color][current];

            current = self.next_vertex[current] as usize;
            if current == index {
                break;
            }
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
        let mut current = index;
        let mut hash = 0;

        loop {
            hash ^= zobrist::TABLE[color][current];
            self.vertices[current] = 0;

            current = self.next_vertex[current] as usize;
            if current == index {
                break;
            }
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

        foreach_4d!(self, index, |other_index, value| {
            if value == opponent && !self.has_n_liberty::<Two>(other_index, 2) {
                adjust ^= self.capture_if(opponent as usize, other_index);
            }
        });

        adjust
    }

    /// 
    #[inline]
    pub fn place(&mut self, color: Color, index: usize) -> u64 {
        let player = color as u8;

        // place the stone on the board regardless of whether it is legal
        // or not.
        self.vertices[index] = color as u8;
        self.next_vertex[index] = index as u16;

        // connect this stone to any neighbouring groups
        foreach_4d!(self, index, |other_index, value| {
            if value == player {
                self.join_vertices(index, other_index);
            }
        });

        // clear the opponents color
        let opponent = color.opposite() as u8;
        let mut hash = zobrist::TABLE[color as usize][index];

        foreach_4d!(self, index, |other_index, value| {
            if value == opponent && !self.has_n_liberty::<One>(other_index, 1) {
                hash ^= self.capture(opponent as usize, other_index);
            }
        });

        hash
    }
}

#[cfg(test)]
mod tests {
    // pass
}
