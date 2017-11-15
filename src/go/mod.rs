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

mod circular_buf;
mod zobrist;

use self::circular_buf::CircularBuf;
use std::fmt;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Color {
    Black = 1,
    White = 2
}

impl Color {
    /// Returns the opposite of this color.
    fn opposite(&self) -> Color {
        match *self {
            Color::Black => Color::White,
            Color::White => Color::Black
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Color::Black => write!(f, "B"),
            Color::White => write!(f, "W")
        }
    }
}

/// Returns an array that contains the mapping from original index to the
/// translated vertex according to the given deltas, if the translated
/// vertex ends up out of bounds then that element is set to 361.
///
/// # Arguments
///
/// * `dx` - the delta in columns
/// * `dy` - the delta in rows
///
fn get_direction_array(dx: i32, dy: i32) -> [usize; 361] {
    let mut out = [0; 361];

    for x in 0..19 {
        let x_ = (x as i32) + dx;

        for y in 0..19 {
            let y_ = (y as i32) + dy;
            let index = 19 * y + x;

            if x_ >= 0 && x_ < 19 && y_ >= 0 && y_ < 19 {
                out[index] = (19 * y_ + x_) as usize;
            } else {
                out[index] = 361;
            }
        }
    }

    out
}

lazy_static! {
    static ref _N: [usize; 361] = get_direction_array(0, 1);
    static ref _E: [usize; 361] = get_direction_array(1, 0);
    static ref _S: [usize; 361] = get_direction_array(0, -1);
    static ref _W: [usize; 361] = get_direction_array(-1, 0);
}

/// Returns `$array[$nested[$index]]` without boundary checks
macro_rules! nested_get_unchecked {
    ($array:expr, $nested:expr, $index:expr) => (unsafe {
        *$array.get_unchecked(*$nested.get_unchecked($index))
    })
}

macro_rules! N { ($array:expr, $index:expr) => (nested_get_unchecked!($array, _N, $index)) }
macro_rules! E { ($array:expr, $index:expr) => (nested_get_unchecked!($array, _E, $index)) }
macro_rules! S { ($array:expr, $index:expr) => (nested_get_unchecked!($array, _S, $index)) }
macro_rules! W { ($array:expr, $index:expr) => (nested_get_unchecked!($array, _W, $index)) }

pub struct Board {
    /// The color of the stone that is occupying each vertex. This array
    /// should in addition contain at least one extra padding element that
    /// contains `0xff`, this extra element is used to the out-of-bounds
    /// index to avoid extra branches.
    vertices: [u8; 368],

    /// The index of a stone that is strongly connected to each vertex in
    /// such a way that every stone in a strongly connected group forms
    /// a cycle.
    next_vertex: [u16; 361],

    /// Stack containing the six most recent `vertices`.
    history: CircularBuf,

    /// The total number of moves that has been played on this board.
    count: u16,

    /// The zobrist hash of the current board state.
    zobrist_hash: u64
}

impl Clone for Board {
    fn clone(&self) -> Board {
        Board {
            vertices: self.vertices,
            next_vertex: self.next_vertex,
            history: self.history.clone(),
            count: self.count,
            zobrist_hash: self.zobrist_hash
        }
    }
}

impl Board {
    /// Returns an empty board state.
    pub fn new() -> Board {
        let mut board = Board {
            vertices: [0; 368],
            next_vertex: [0; 361],
            history: CircularBuf::new(),
            count: 0,
            zobrist_hash: 0
        };

        for i in 361..368 {
            board.vertices[i] = 0xff;
        }

        board
    }

    /// Returns the width and height of this board.
    #[inline]
    pub fn size(&self) -> usize {
        19
    }

    /// Returns the zobrist hash of this board.
    #[inline]
    pub fn zobrist_hash(&self) -> u64 {
        self.zobrist_hash
    }

    /// Returns the color (if the vertex is not empty) of the stone at
    /// the given coordinates.
    ///
    /// # Arguments
    ///
    /// * `x` - the column of the coordinates
    /// * `y` - the row of the coordinates
    ///
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> Option<Color> {
        let index = 19 * y + x;

        if self.vertices[index] == Color::Black as u8 {
            Some(Color::Black)
        } else if self.vertices[index] == Color::White as u8 {
            Some(Color::White)
        } else {
            None
        }
    }

    /// Returns true iff the group at the given index at least one liberty.
    ///
    /// # Arguments
    ///
    /// * `index` - the index of a stone in the group to check
    ///
    fn has_one_liberty(&self, index: usize) -> bool {
        let mut current = index;

        loop {
            if N!(self.vertices, current) == 0 { return true; }
            if E!(self.vertices, current) == 0 { return true; }
            if S!(self.vertices, current) == 0 { return true; }
            if W!(self.vertices, current) == 0 { return true; }

            current = self.next_vertex[current] as usize;
            if current == index {
                break
            }
        }

        false
    }

    /// Returns true iff the group at the given index has at least two
    /// liberties.
    ///
    /// # Arguments
    ///
    /// * `index` - the index of a stone in the group to check
    ///
    fn has_two_liberties(&self, index: usize) -> bool {
        let mut current = index;
        let mut previous = 0xffff;

        loop {
            macro_rules! check_two_liberties {
                ($index:expr) => ({
                    if previous != 0xffff && previous != $index {
                        return true;
                    } else {
                        previous = $index;
                    }
                })
            }

            if N!(self.vertices, current) == 0 { check_two_liberties!(current + 19); }
            if E!(self.vertices, current) == 0 { check_two_liberties!(current + 1); }
            if S!(self.vertices, current) == 0 { check_two_liberties!(current - 19); }
            if W!(self.vertices, current) == 0 { check_two_liberties!(current - 1); }

            current = self.next_vertex[current] as usize;
            if current == index {
                break
            }
        }

        false
    }

    /// Remove all stones strongly connected to the given index from the board.
    ///
    /// # Arguments
    ///
    /// * `index` - the index of a stone in the group to capture
    ///
    fn capture(&mut self, index: usize) {
        let mut current = index;

        loop {
            let c = self.vertices[current] as usize;

            self.zobrist_hash ^= zobrist::TABLE[c][current];
            self.vertices[current] = 0;

            current = self.next_vertex[current] as usize;
            if current == index {
                break
            }
        }
    }

    /// Remove all stones strongly connected to the given index from the given array
    /// using the group definition from this board.
    ///
    /// # Arguments
    ///
    /// * `index` - the index of a stone in the group to capture
    ///
    fn capture_other(&self, vertices: &mut [u8], index: usize) {
        let mut current = index;

        loop {
            vertices[current] = 0;

            current = self.next_vertex[current] as usize;
            if current == index {
                break
            }
        }
    }

    /// Connects the chains of the two vertices into one chain. This method
    /// should not be called with the same group twice as that will result
    /// in a corrupted chain.
    ///
    /// # Arguments
    ///
    /// * `index` - The first chain to connect
    /// * `other` - The second chain to connect
    ///
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

        // re-connect the two lists so if we have two chains A and B:
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

    /// Returns whether the given move is valid according to the
    /// Tromp-Taylor rules.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `index` - the HW index of the move
    ///
    pub fn _is_valid(&self, color: Color, index: usize) -> bool {
        self.vertices[index] == 0 && {
            let n = N!(self.vertices, index);
            let e = E!(self.vertices, index);
            let s = S!(self.vertices, index);
            let w = W!(self.vertices, index);

            // check for direct liberties
            if n == 0 { return true; }
            if e == 0 { return true; }
            if s == 0 { return true; }
            if w == 0 { return true; }

            // check for the following two conditions simplied into one case:
            //
            // 1. If a neighbour is friendly then we are fine if it has at
            //    least two liberties.
            // 2. If a neighbour is unfriendly then we are fine if it has less
            //    than two liberties (i.e. one).
            let current = color as u8;

            if n != 0xff && (n == current) == self.has_two_liberties(index + 19) { return true; }
            if e != 0xff && (e == current) == self.has_two_liberties(index + 1) { return true; }
            if s != 0xff && (s == current) == self.has_two_liberties(index - 19) { return true; }
            if w != 0xff && (w == current) == self.has_two_liberties(index - 1) { return true; }

            false  // move is suicide :'(
        }
    }

    /// Returns whether the given move is valid according to the
    /// Tromp-Taylor rules.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `x` - the column of the move
    /// * `y` - the row of the move
    ///
    pub fn is_valid(&self, color: Color, x: usize, y: usize) -> bool {
        self._is_valid(color, 19 * y + x)
    }

    /// Place the given stone on the board without checking if it is legal, the
    /// board is then updated according to the Tromp-Taylor rules with the
    /// except that ones own color is not cleared.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the move
    /// * `x` - The column of the move
    /// * `y` - The row of the move
    ///
    pub fn place(&mut self, color: Color, x: usize, y: usize) {
        let index = 19 * y + x;

        // place the stone on the board regardless of whether it is legal
        // or not.
        self.vertices[index] = color as u8;
        self.next_vertex[index] = index as u16;
        self.count += 1;
        self.zobrist_hash ^= zobrist::TABLE[color as usize][index];

        // connect this stone to any neighbouring groups
        let player = color as u8;

        if N!(self.vertices, index) == player { self.join_vertices(index, index + 19); }
        if E!(self.vertices, index) == player { self.join_vertices(index, index + 1); }
        if S!(self.vertices, index) == player { self.join_vertices(index, index - 19); }
        if W!(self.vertices, index) == player { self.join_vertices(index, index - 1); }

        // clear the opponents color
        let opponent = color.opposite() as u8;

        if N!(self.vertices, index) == opponent && !self.has_one_liberty(index + 19) { self.capture(index + 19); }
        if E!(self.vertices, index) == opponent && !self.has_one_liberty(index + 1) { self.capture(index + 1); }
        if S!(self.vertices, index) == opponent && !self.has_one_liberty(index - 19) { self.capture(index - 19); }
        if W!(self.vertices, index) == opponent && !self.has_one_liberty(index - 1) { self.capture(index - 1); }

        // add the current board state to the history *after* we have updated it because:
        //
        // 1. that way we do not need a special case to retrieve the current board when
        //    generating features.
        // 2. the circular stack starts with all buffers as zero, so there is no need to
        //    keep track of the initial board state.
        self.history.push(&self.vertices);
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
    fn fill_liberties(&self, vertices: &[u8], index: usize, liberties: &mut [u8]) {
        let mut current = index;

        loop {
            liberties[_N[current]] = N!(vertices, current);
            liberties[_E[current]] = E!(vertices, current);
            liberties[_S[current]] = S!(vertices, current);
            liberties[_W[current]] = W!(vertices, current);

            current = self.next_vertex[current] as usize;
            if current == index {
                break
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
    /// * `index` - the index of the group to check
    /// * `memoize` - cache of already calculated liberty counts
    ///
    fn get_num_liberties(&self, index: usize, memoize: &mut [usize]) -> usize {
        if memoize[index] != 0 {
            memoize[index]
        } else {
            let mut liberties = [0xff; 368];

            self.fill_liberties(&self.vertices, index, &mut liberties);

            // count the number of liberties, maybe in the future using a SIMD
            // implementation which would be a lot faster than this
            let num_liberties = (0..361).filter(|i| liberties[*i] == 0).count();

            // update the cached value in the memoize array for all stones
            // that are strongly connected to the given index
            let mut current = index;

            loop {
                memoize[current] = num_liberties;

                current = self.next_vertex[current] as usize;
                if current == index {
                    break
                }
            }

            num_liberties
        }
    }

    /// Returns the number of liberties of the group connected to the given stone
    /// *if* it was played, will panic if the vertex is not empty.
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the stone to pretend place
    /// * `index` - the index of the stone to pretend place
    ///
    fn get_num_liberties_if(&self, color: Color, index: usize) -> usize {
        debug_assert!(self.vertices[index] == 0);

        let mut vertices = self.vertices.clone();

        vertices[index] = color as u8;

        // capture of opponent stones 
        let current = color as u8;
        let opponent = color.opposite() as u8;

        if N!(vertices, index) == opponent && !self.has_two_liberties(index + 19) { self.capture_other(&mut vertices, index + 19); }
        if E!(vertices, index) == opponent && !self.has_two_liberties(index + 1) { self.capture_other(&mut vertices, index + 1); }
        if S!(vertices, index) == opponent && !self.has_two_liberties(index - 19) { self.capture_other(&mut vertices, index - 19); }
        if W!(vertices, index) == opponent && !self.has_two_liberties(index - 1) { self.capture_other(&mut vertices, index - 1); }

        // add liberties based on the liberties of the friendly neighbouring
        // groups
        let mut liberties = [0xff; 368];

        if N!(vertices, index) == current { self.fill_liberties(&vertices, index + 19, &mut liberties); }
        if E!(vertices, index) == current { self.fill_liberties(&vertices, index + 1, &mut liberties); }
        if S!(vertices, index) == current { self.fill_liberties(&vertices, index - 19, &mut liberties); }
        if W!(vertices, index) == current { self.fill_liberties(&vertices, index - 1, &mut liberties); }

        // add direct liberties of the new stone
        liberties[_N[index]] = N!(vertices, index);
        liberties[_E[index]] = E!(vertices, index);
        liberties[_S[index]] = S!(vertices, index);
        liberties[_W[index]] = W!(vertices, index);

        (0..361).filter(|i| liberties[*i] == 0).count()
    }

    /// Returns an array containing the (manhattan) distance to the closest stone
    /// of the given color for each point on the board.
    /// 
    /// # Arguments
    /// 
    /// * `color` - the color to get the distance from
    /// 
    fn get_territory_distance(&self, color: Color) -> [u8; 368] {
        let current = color as u8;

        // find all of our stones and mark them as starting points
        let mut territory = [0xff; 368];
        let mut probes = vec! [];

        for index in 0..361 {
            if self.vertices[index] == current as u8 {
                territory[index] = 0;
                probes.push(index);
            }
        }

        // x
        while probes.len() > 0 {
            let index = probes.pop().unwrap();
            let t = territory[index] + 1;

            if N!(self.vertices, index) == 0 && N!(territory, index) > t { probes.push(_N[index]); territory[_N[index]] = t; }
            if E!(self.vertices, index) == 0 && E!(territory, index) > t { probes.push(_E[index]); territory[_E[index]] = t; }
            if S!(self.vertices, index) == 0 && S!(territory, index) > t { probes.push(_S[index]); territory[_S[index]] = t; }
            if W!(self.vertices, index) == 0 && W!(territory, index) > t { probes.push(_W[index]); territory[_W[index]] = t; }
        }

        territory
    }

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
    /// 15. Our territory (closest to our vertex)
    /// 16. Our vertices (now)
    /// 17. Our vertices (now-1)
    /// 18. Our vertices (now-2)
    /// 19. Our vertices (now-3)
    /// 20. Our vertices (now-4)
    /// 21. Our vertices (now-5)
    /// 22. Opponent liberties (1)
    /// 23. Opponent liberties (2)
    /// 24. Opponent liberties (3)
    /// 25. Opponent liberties (4)
    /// 26. Opponent liberties (5)
    /// 27. Opponent liberties (6+)
    /// 28. Opponent territory (closest to opponent vertex)
    /// 29. Opponent vertices (now)
    /// 30. Opponent vertices (now-1)
    /// 31. Opponent vertices (now-2)
    /// 32. Opponent vertices (now-3)
    /// 33. Opponent vertices (now-4)
    /// 34. Opponent vertices (now-5)
    ///
    /// # Arguments
    ///
    /// * `color` - the color of the current player
    ///
    pub fn get_features(&self, color: Color) -> Box<[f32]> {
        let mut features = vec! [ 0.0f32; 34 * 361 ];
        let is_black = if color == Color::Black { 1.0 } else { 0.0 };
        let current = color as u8;

        // set the two constant planes and the liberties
        let mut liberties = [0; 368];

        for index in 0..361 {
            features[0 * 361 + index] = 1.0;
            features[1 * 361 + index] = is_black;

            if self.vertices[index] != 0 {
                let num_liberties = ::std::cmp::min(
                    self.get_num_liberties(index, &mut liberties),
                    6
                );
                let l = {
                    debug_assert!(num_liberties > 0);

                    if self.vertices[index] == current {
                        1 + num_liberties
                    } else {
                        20 + num_liberties
                    }
                };

                features[l * 361 + index] = 1.0;
            } else if self._is_valid(color, index) {
                let num_liberties = ::std::cmp::min(
                    self.get_num_liberties_if(color, index),
                    6
                );
                let l = 7 + num_liberties;

                features[l * 361 + index] = 1.0;
            }
        }

        // set the territory planes
        let our_territory = self.get_territory_distance(color);
        let opponent_territory = self.get_territory_distance(color.opposite());

        for index in 0..361 {
            if self.vertices[index] != 0 {
                // pass
            } else if our_territory[index] < opponent_territory[index] {
                features[14 * 361 + index] = 1.0;
            } else if opponent_territory[index] < our_territory[index] {
                features[27 * 361 + index] = 1.0;
            }
        }

        // set the 12 planes that denotes our and the opponents stones
        for (i, vertices) in self.history.iter().enumerate() {
            for index in 0..361 {
                if vertices[index] == 0 {
                    // pass
                } else if vertices[index] == current {
                    let p = 15 + i;

                    features[p * 361 + index] = 1.0;
                } else { // opponent
                    let p = 28 + i;

                    features[p * 361 + index] = 1.0;
                }
            }
        }

        features.into_boxed_slice()
    }
}

impl fmt::Display for Board {
    /// Pretty-print the current board in a similar format as the KGS client.
    ///
    /// # Arguments
    ///
    /// * `f` - the formatter to write the game to
    ///
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const LETTERS: [char; 25] = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z'
        ];

        write!(f, "    ")?;
        for i in 0..19 { write!(f, " {}", LETTERS[i])?; }
        write!(f, "\n")?;
        write!(f, "   \u{256d}")?;
        for _ in 0..19 { write!(f, "\u{2500}\u{2500}")?; }
        write!(f, "\u{2500}\u{256e}\n")?;

        for y in 0..19 {
            let y = 18 - y;

            write!(f, "{:2} \u{2502}", 1 + y)?;

            for x in 0..19 {
                let index = 19 * y + x;

                if self.vertices[index] == 0 {
                    write!(f, "  ")?;
                } else if self.vertices[index] == Color::Black as u8 {
                    write!(f, " \u{25cf}")?;
                } else if self.vertices[index] == Color::White as u8 {
                    write!(f, " \u{25cb}")?;
                }
            }

            write!(f, " \u{2502} {}\n", 1 + y)?;
        }

        write!(f, "   \u{2570}")?;
        for _ in 0..19 { write!(f, "\u{2500}\u{2500}")?; }
        write!(f, "\u{2500}\u{256f}\n")?;
        write!(f, "    ")?;
        for i in 0..19 { write!(f, " {}", LETTERS[i])?; }
        write!(f, "\n")?;
        write!(f, "    \u{25cf} Black    \u{25cb} White\n")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use go::*;

    /// Test that it is possible to capture a stone in the middle of the
    /// board.
    #[test]
    fn capture() {
        let mut board = Board::new();

        board.place(Color::Black,  9,  9);
        board.place(Color::White,  8,  9);
        board.place(Color::White, 10,  9);
        board.place(Color::White,  9,  8);
        board.place(Color::White,  9, 10);

        assert_eq!(board.at(9, 9), None);
    }

    /// Test that it is possible to capture a group of stones in the corner.
    #[test]
    fn capture_group() {
        let mut board = Board::new();

        board.place(Color::Black, 0, 1);
        board.place(Color::Black, 1, 0);
        board.place(Color::Black, 0, 0);
        board.place(Color::Black, 1, 1);

        board.place(Color::White, 2, 0);
        board.place(Color::White, 2, 1);
        board.place(Color::White, 0, 2);
        board.place(Color::White, 1, 2);

        assert_eq!(board.at(0, 0), None);
        assert_eq!(board.at(0, 1), None);
        assert_eq!(board.at(1, 0), None);
        assert_eq!(board.at(1, 1), None);
    }

    /// Test that it is not possible to play a suicide move in the corner
    /// with two adjacent neighbours of the opposite color.
    #[test]
    fn suicide_corner() {
        let mut board = Board::new();

        board.place(Color::White, 0, 0);
        board.place(Color::Black, 1, 0);
        board.place(Color::Black, 0, 1);

        assert_eq!(board.at(0, 0), None);
        assert!(!board.is_valid(Color::White, 0, 0));
        assert!(board.is_valid(Color::Black, 0, 0));
    }

    /// Test that it is not possible to play a suicide move in the middle
    /// of a ponnuki.
    #[test]
    fn suicide_middle() {
        let mut board = Board::new();

        board.place(Color::Black,  9,  9);
        board.place(Color::White,  8,  9);
        board.place(Color::White, 10,  9);
        board.place(Color::White,  9,  8);
        board.place(Color::White,  9, 10);

        assert_eq!(board.at(9, 9), None);
        assert!(!board.is_valid(Color::Black, 9, 9));
        assert!(board.is_valid(Color::White, 9, 9));
    }

    /// Test so that the correct number of pretend liberties are correct.
    #[test]
    fn liberties_if() {
        let mut board = Board::new();

        board.place(Color::White, 0, 0);
        board.place(Color::Black, 0, 1);
        board.place(Color::Black, 1, 1);

        assert_eq!(board.get_num_liberties_if(Color::Black, 1), 5);
    }
}
