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

use go::asm;
use go::board_fast::*;
use go::board::Board;
use go::color::Color;
use go::ladder::Ladder;
use go::symmetry;

use std::collections::VecDeque;

/// Utility function for determining the data format of the array returned by
/// `get_features`.
pub trait Order {
    fn index(c: usize, i: usize) -> usize;
}

/// Implementation of `Order` for the data format `NCHW`.
pub struct CHW;

impl Order for CHW {
    fn index(c: usize, i: usize) -> usize {
        c * 361 + i
    }
}

/// Implementation of `Order` for the data format `NHWC`.
pub struct HWC;

impl Order for HWC {
    fn index(c: usize, i: usize) -> usize {
        i * 36 + c
    }
}

pub trait Features {
    /// Returns the features of the current object in the given order and data
    /// type.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the current player
    /// * `symmetry` - the symmetry to use
    ///
    fn get_features<T: From<f32> + Copy, O: Order>(
        &self,
        color: Color,
        symmetry: symmetry::Transform
    ) -> Vec<T>;
}

impl Features for Board {
    /// Returns the features of the current board state for the given color,
    /// it returns the following features:
    ///
    /// 1. A constant plane filled with ones
    /// 2. A constant plane filled with ones if we are black
    /// 3. Our liberties (1)
    /// 4. Our liberties (2)
    /// 5. Our liberties (3)
    /// 6. Our liberties (4)
    /// 7. Our liberties (5)
    /// 8. Our liberties (6+)
    /// 9. Our liberties after move (1)
    /// 10. Our liberties after move (2)
    /// 11. Our liberties after move (3)
    /// 12. Our liberties after move (4)
    /// 13. Our liberties after move (5)
    /// 14. Our liberties after move (6+)
    /// 15. Our vertices (now)
    /// 16. Our vertices (now-1)
    /// 17. Our vertices (now-2)
    /// 18. Our vertices (now-3)
    /// 19. Our vertices (now-4)
    /// 20. Our vertices (now-5)
    /// 21. Opponent liberties (1)
    /// 22. Opponent liberties (2)
    /// 23. Opponent liberties (3)
    /// 24. Opponent liberties (4)
    /// 25. Opponent liberties (5)
    /// 26. Opponent liberties (6+)
    /// 27. Opponent vertices (now)
    /// 28. Opponent vertices (now-1)
    /// 29. Opponent vertices (now-2)
    /// 30. Opponent vertices (now-3)
    /// 31. Opponent vertices (now-4)
    /// 32. Opponent vertices (now-5)
    /// 33. Is ladder capture
    /// 34. Is ladder escape
    /// 35. Is point only reachable from our vertices
    /// 36. Is point only reachable from opponent vertices
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the current player
    ///
    fn get_features<T: From<f32> + Copy, O: Order>(
        &self,
        color: Color,
        symmetry: symmetry::Transform
    ) -> Vec<T>
    {
        let c_0: T = T::from(0.0);
        let c_1: T = T::from(1.0);

        let mut features = vec! [c_0; 36 * 361];
        let symmetry_table = symmetry.get_table();
        let is_black = if color == Color::Black { c_1 } else { c_0 };
        let current = color as u8;

        // set the two constant planes and the liberties
        let mut liberties = [0; 368];

        for index in 0..361 {
            let other = symmetry_table[index] as usize;

            features[O::index(0, other)] = c_1;
            features[O::index(1, other)] = is_black;

            if self.inner.vertices[index] != 0 {
                // num liberties
                let num_liberties = ::std::cmp::min(
                    get_num_liberties(&self.inner, index, &mut liberties),
                    6
                );
                let l = {
                    debug_assert!(num_liberties > 0);

                    if self.inner.vertices[index] == current {
                        1 + num_liberties
                    } else {
                        19 + num_liberties
                    }
                };

                features[O::index(l, other)] = c_1;
            } else if _is_valid_memoize(&self.inner, color, index, &mut liberties) {
                // liberties after move
                let num_liberties = ::std::cmp::min(
                    get_num_liberties_if(&self.inner, color, index, &mut liberties),
                    6
                );
                let l = 7 + num_liberties;

                features[O::index(l, other)] = c_1;

                // is ladder capture
                if self.inner.is_ladder_capture(color, index) {
                    features[O::index(32, other)] = c_1;
                }

                // is ladder escape
                if self.inner.is_ladder_escape(color, index) {
                    features[O::index(33, other)] = c_1;
                }
            }
        }

        // set the 12 planes that denotes our and the opponents stones
        for (i, vertices) in self.history.iter().enumerate() {
            for index in 0..361 {
                let other = symmetry_table[index] as usize;

                if vertices[index] == 0 {
                    // pass
                } else if vertices[index] == current {
                    let p = 14 + i;

                    features[O::index(p, other)] = c_1;
                } else {
                    // opponent
                    let p = 26 + i;

                    features[O::index(p, other)] = c_1;
                }
            }
        }

        // set the territory features
        let our_territory = get_territory_distance(&self.inner, color);
        let other_territory = get_territory_distance(&self.inner, color.opposite());

        for index in 0..361 {
            let other = symmetry_table[index] as usize;

            if our_territory[index] > 0 && other_territory[index] == 0xff {
                features[O::index(34, other)] = c_1;
            } else if our_territory[index] == 0xff && other_territory[index] > 0 {
                features[O::index(35, other)] = c_1;
            }
        }

        features
    }
}

/// Fills the given array with all liberties of in the provided array of vertices
/// for the group.
///
/// # Arguments
///
/// * `vertices` - the array to fill liberties from
/// * `index` - the group to fill liberties for
/// * `liberties` - output array containing the liberties of this group
///
fn fill_liberties(board: &BoardFast, index: usize, liberties: &mut [u8]) {
    let mut current = index;

    loop {
        foreach_4d!(board, current, |other_index, value| {
            unsafe {
                *liberties.get_unchecked_mut(other_index) = value;
            }
        });

        current = board.next_vertex[current] as usize;

        if current == index {
            break;
        }
    }
}

/// Returns the number of liberties of the given group using any recorded
/// value in `memoize` if available otherwise it is calculated. Any
/// calculated value is written back to `memoize` for all strongly
/// connected stones.
///
/// # Arguments
///
/// * `board` - 
/// * `index` - the index of the group to check
/// * `memoize` - cache of already calculated liberty counts
///
fn get_num_liberties(board: &BoardFast, index: usize, memoize: &mut [usize]) -> usize {
    if memoize[index] != 0 {
        memoize[index]
    } else {
        let mut liberties = [0xff; 384];

        fill_liberties(board, index, &mut liberties);

        // count the number of liberties, maybe in the future using a SIMD
        // implementation which would be a lot faster than this
        let num_liberties = asm::count_zeros(&liberties);

        // update the cached value in the memoize array for all stones
        // that are strongly connected to the given index
        let mut current = index;

        loop {
            memoize[current] = num_liberties;

            current = board.next_vertex[current] as usize;
            if current == index {
                break;
            }
        }

        num_liberties
    }
}

/// Returns whether the given move is valid according to the
/// Tromp-Taylor rules using the provided `memoize` table to
/// determine the number of liberties.
///
/// This function also assume the given vertex is empty and does
/// not perform the check itself.
///
/// # Arguments
///
/// * `color` - the color of the move
/// * `index` - the HW index of the move
/// * `memoize` - cache of already calculated liberty counts
///
fn _is_valid_memoize(board: &BoardFast, color: Color, index: usize, memoize: &mut [usize]) -> bool {
    debug_assert!(board.vertices[index] == 0);

    let current = color as u8;

    foreach_4d!(board, index, |other_index, value| {
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
        if value != 0xff && (value == current) == (get_num_liberties(board, other_index, memoize) >= 2) {
            return true;
        }
    });

    false  // move is suicide :'(
}

/// Returns the number of liberties of the group connected to the given stone
/// *if* it was played, will panic if the vertex is not empty.
///
/// # Arguments
///
/// * `color` - the color of the stone to pretend place
/// * `index` - the index of the stone to pretend place
///
fn get_num_liberties_if(board: &BoardFast, color: Color, index: usize, memoize: &mut [usize]) -> usize {
    debug_assert!(board.vertices[index] == 0);

    let mut other = board.clone();
    other.vertices[index] = color as u8;

    // capture of opponent stones
    let current = color as u8;
    let opponent = color.opposite() as u8;

    foreach_4d!(board, index, |other_index, value| {
        if value == opponent && get_num_liberties(&board, other_index, memoize) == 1 {
            other.capture(opponent as usize, other_index);
        }
    });

    // add liberties based on the liberties of the friendly neighbouring
    // groups
    let mut liberties = [0xff; 384];

    foreach_4d!(other, index, |other_index, value| {
        if value == current {
            fill_liberties(&other, other_index, &mut liberties);
        }

        // add direct liberties of the new stone
        liberties[other_index] = value;
    });

    asm::count_zeros(&liberties)
}

/// Returns an array containing the (manhattan) distance to the closest stone
/// of the given color for each point on the board.
///
/// # Arguments
///
/// * `color` - the color to get the distance from
///
fn get_territory_distance(board: &BoardFast, color: Color) -> [u8; 368] {
    let current = color as u8;

    // find all of our stones and mark them as starting points
    let mut territory = [0xff; 368];
    let mut probes = VecDeque::with_capacity(512);

    for index in 0..361 {
        if board.vertices[index] == current {
            territory[index] = 0;
            probes.push_back(index);
        }
    }

    // compute the distance to all neighbours using a dynamic programming
    // approach where we at each iteration try to update the neighbours of
    // each updated vertex, and if the distance we tried to set was smaller
    // than the current distance we try to update that vertex neighbours.
    //
    // This is equivalent to a Bellman–Ford algorithm for the shortest path.
    while !probes.is_empty() {
        let index = probes.pop_front().unwrap();
        let t = territory[index] + 1;

        foreach_4d!(board, index, |other_index, value| {
            if value == 0 && territory[other_index] > t {
                probes.push_back(other_index);
                territory[other_index] = t;
            }
        });
    }

    territory
}

#[cfg(test)]
mod tests {
    // pass
}
