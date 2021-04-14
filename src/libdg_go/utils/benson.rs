// Copyright 2020 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use crate::board_fast::BoardFast;
use crate::point_state::Vertex;
use crate::iter::{AdjacentChainIter, ChainIter, ValidIter, HasColor};
use crate::utils::flood_fill::FloodFill;
use crate::{Board, Color, Point};

pub trait Block {
    type PointIter: Iterator<Item=Point>;

    fn points(&self) -> Self::PointIter;
    fn is_liberty(&self, point: Point) -> bool;
}

pub trait AllBlocks<'a> {
    type Block: Block + Sized;

    fn all(board: &'a BoardFast, to_move: Color) -> Vec<Self::Block>;
}

pub trait Region {
    type NeighbourIter: Iterator<Item=Point>;
    type PointIter: Iterator<Item=Point>;

    fn neighbours(&self) -> Self::NeighbourIter;
    fn points(&self) -> Self::PointIter;
}

pub trait AllRegions<'a> {
    type Region: Region + Sized;

    fn all(board: &'a BoardFast, to_move: Color) -> Vec<Self::Region>;
}

#[derive(Clone, Copy, PartialEq)]
enum PointStatus {
    None,
    Block,
    Region,
}

pub struct Benson<'a, R: AllRegions<'a>, B: AllBlocks<'a>> {
    regions: Vec<R::Region>,
    blocks: Vec<B::Block>,
    points: [PointStatus; Point::MAX],
}

impl<'a, R: AllRegions<'a>, B: AllBlocks<'a>> Benson<'a, R, B> {
    pub fn new(board: &'a Board, to_move: Color) -> Self {
        let mut out = Self {
            regions: R::all(&board.inner, to_move),
            blocks: B::all(&board.inner, to_move),
            points: [PointStatus::None; Point::MAX]
        };

        out.mark_all_blocks();
        out.mark_all_regions();
        out.remove_non_vital_regions();
        while out.remove_non_alive_blocks() | out.remove_non_surrounded_regions() {
            // pass
        }

        out
    }

    /// Mark all blocks in `blocks` into `points`.
    fn mark_all_blocks(&mut self) {
        for block in &self.blocks {
            mark_points(&mut self.points, block.points(), PointStatus::Block);
        }
    }

    /// Mark all regions in `regions` into `points`.
    fn mark_all_regions(&mut self) {
        for region in &self.regions {
            mark_points(&mut self.points, region.points(), PointStatus::Region);
        }
    }

    /// Remove from X all blocks with fewer than two healthy enclosed regions
    /// in R.
    fn remove_non_alive_blocks(&mut self) -> bool {
        let points = &mut self.points;
        let blocks = &mut self.blocks;
        let original_len = blocks.len();
        let regions = &self.regions;

        blocks.retain(|b| {
            let is_alive = regions.iter().filter(|&r| is_vital(r, b)).count() >= 2;

            if !is_alive {
                mark_points(points, b.points(), PointStatus::None);
            }

            is_alive
        });

        blocks.len() < original_len
    }

    /// Remove from R all enclosed regions with a surrounding stone in a block
    /// not in X.
    fn remove_non_surrounded_regions(&mut self) -> bool {
        let points = &mut self.points;
        let regions = &mut self.regions;
        let original_len = regions.len();

        regions.retain(|r| {
            let is_healthy = r.neighbours().all(|p| points[p.to_i()] == PointStatus::Block);

            if !is_healthy {
                mark_points(points, r.points(), PointStatus::None);
            }

            is_healthy
        });

        regions.len() < original_len
    }

    /// Remove from R all enclosed regions that are not vital to at least one
    /// block in X.
    fn remove_non_vital_regions(&mut self) {
        let blocks = &self.blocks;
        let points = &mut self.points;
        let regions = &mut self.regions;

        regions.retain(|r| {
            let is_vital = blocks.iter().any(|b| is_vital(r, b));

            if !is_vital {
                mark_points(points, r.points(), PointStatus::None);
            }

            is_vital
        });
    }

    /// Returns if the given `point` is part of an unconditionally alive block
    ///
    /// # Arguments
    ///
    /// * `point` -
    ///
    pub fn is_alive(&self, point: Point) -> bool {
        self.points[point.to_i()] == PointStatus::Block
    }

    /// Returns if the given `point` is vital to an unconditionally alive
    /// block.
    ///
    /// # Arguments
    ///
    /// * `point` -
    ///
    pub fn is_eye(&self, point: Point) -> bool {
        self.points[point.to_i()] == PointStatus::Region
    }

    /// Returns if the given `point` is not a liberty of a group that is
    /// unconditionally alive.
    ///
    /// # Arguments
    ///
    /// * `point` -
    ///
    pub fn is_valid(&self, point: Point) -> bool {
        self.points[point.to_i()] == PointStatus::None
    }
}

/// Set all `points` in the iterator to `value`.
///
/// # Arguments
///
/// * `points` -
/// * `iter` -
/// * `value` -
///
fn mark_points<I: Iterator<Item=Point>>(points: &mut [PointStatus], iter: I, value: PointStatus) {
    for point in iter {
        points[point.to_i()] = value;
    }
}

/// Return if all points in the given `region` is a liberty of the given
/// `block`.
///
/// # Arguments
///
/// * `region` -
/// * `block` -
///
fn is_vital<B: Block, R: Region>(region: &R, block: &B) -> bool {
    region.points().all(|p| block.is_liberty(p))
}

pub struct BlockImpl<'a> {
    board: &'a BoardFast,
    block_at: Point,
}

impl<'a> Block for BlockImpl<'a> {
    type PointIter = ChainIter<*const BoardFast>;

    fn points(&self) -> Self::PointIter {
        self.board.block_at(self.block_at).into_iter()
    }

    fn is_liberty(&self, point: Point) -> bool {
        self.board.adjacencies_of(self.block_at).any(|l| l == point)
    }
}

pub struct AllBlocksImpl;

impl<'a> AllBlocks<'a> for AllBlocksImpl {
    type Block = BlockImpl<'a>;

    fn all(board: &'a BoardFast, to_move: Color) -> Vec<BlockImpl<'a>> {
        let mut visited = vec! [false; Point::MAX];

        Point::all()
            .filter(|&p| {
                board[p].color() == Some(to_move) && {
                    let head = board[p].head_point();
                    let has_visit = visited[head.to_i()];
                    visited[head.to_i()] = true;

                    !has_visit
                }
            })
            .map(|block_at| BlockImpl { board, block_at })
            .collect::<Vec<_>>()
    }
}

pub struct UnsafeSliceIter<T> {
    ptr: *const T,
    len: usize,
    pos: usize,
}

impl<T: Clone + Copy> Iterator for UnsafeSliceIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.len {
            let out = unsafe { *self.ptr.add(self.pos) };
            self.pos += 1;

            Some(out)
        } else {
            None
        }
    }
}

pub struct RegionImpl {
    points: Vec<Point>,
    neighbours: Vec<Point>,
}

impl Region for RegionImpl {
    type NeighbourIter = UnsafeSliceIter<Point>;
    type PointIter = UnsafeSliceIter<Point>;

    fn neighbours(&self) -> Self::NeighbourIter {
        UnsafeSliceIter {
            ptr: self.neighbours.as_ptr(),
            len: self.neighbours.len(),
            pos: 0
        }
    }

    fn points(&self) -> Self::PointIter {
        UnsafeSliceIter {
            ptr: self.points.as_ptr(),
            len: self.points.len(),
            pos: 0
        }
    }
}

pub struct AllRegionsImpl;

impl<'a> AllRegions<'a> for AllRegionsImpl {
    type Region = RegionImpl;

    fn all(board: &'a BoardFast, to_move: Color) -> Vec<Self::Region> {
        let flood =
            FloodFill::new(
                board,
                |board, point| board[point].color() == None,
                |board, point| board[point].color() == Some(to_move)
            );

        flood.starting_points()
            .filter_map(|&starting_point| {
                let points = flood.region_at(starting_point).collect::<Vec<_>>();
                let neighbours =
                    ValidIter::new(
                        AdjacentChainIter::new(flood.region_at(starting_point)),
                        HasColor::new(board, Some(to_move))
                    )
                    .collect::<Vec<_>>();

                if neighbours.len() == 0 {
                    None
                } else {
                    Some(RegionImpl { points, neighbours })
                }
            })
            .collect::<Vec<_>>()
    }
}

pub type BensonImpl<'a> = Benson<'a, AllRegionsImpl, AllBlocksImpl>;

#[cfg(test)]
mod tests {
    use test::Bencher;
    use super::*;

    struct PointIter {
        array: *const Point,
        len: usize,
        count: usize
    }

    impl Iterator for PointIter {
        type Item = Point;

        fn next(&mut self) -> Option<Point> {
            if self.count < self.len {
                let out = unsafe { *(self.array.add(self.count)) };
                self.count += 1;

                Some(out)
            } else {
                None
            }
        }
    }

    struct FakeBlock {
        points: Vec<Point>,
        liberties: Vec<Point>,
    }

    impl Block for FakeBlock {
        type PointIter = PointIter;

        fn points(&self) -> Self::PointIter {
            PointIter { array: self.points.as_ptr(), len: self.points.len(), count: 0 }
        }

        fn is_liberty(&self, point: Point) -> bool {
            self.liberties.contains(&point)
        }
    }


    struct FakeAllBlocks;

    impl<'a> AllBlocks<'a> for FakeAllBlocks {
        type Block = FakeBlock;

        fn all(_board: &'a BoardFast, _to_move: Color) -> Vec<Self::Block> {
            vec! [
                FakeBlock {  // a1
                    points: vec! [
                        Point::new(0, 0),
                        Point::new(0, 1),
                        Point::new(0, 2),
                        Point::new(0, 3),
                        Point::new(1, 0),
                        Point::new(1, 3),
                        Point::new(1, 4),
                        Point::new(2, 4),
                        Point::new(2, 5),
                    ],
                    liberties: vec! [
                        Point::new(1, 1),
                        Point::new(1, 2),
                        Point::new(2, 3),
                        Point::new(3, 4),
                        Point::new(3, 5),
                        Point::new(1, 5),
                        Point::new(0, 4),
                    ],
                },
                FakeBlock {  // a6
                    points: vec! [
                        Point::new(0, 5)
                    ],
                    liberties: vec! [
                        Point::new(0, 4),
                        Point::new(1, 5),
                    ]
                },
                FakeBlock {  // c2
                    points: vec! [
                        Point::new(2, 1),
                    ],
                    liberties: vec! [
                        Point::new(2, 0),
                        Point::new(2, 2),
                        Point::new(1, 1),
                        Point::new(3, 1),
                    ],
                },
                FakeBlock {  // e6
                    points: vec! [
                        Point::new(4, 5),
                    ],
                    liberties: vec! [
                        Point::new(3, 5),
                        Point::new(5, 5),
                        Point::new(4, 4),
                    ]
                },
                FakeBlock {  // d1
                    points: vec! [
                        Point::new(3, 0),
                        Point::new(4, 0),
                        Point::new(4, 1),
                        Point::new(5, 1),
                        Point::new(5, 2),
                        Point::new(5, 3),
                        Point::new(5, 4),
                    ],
                    liberties: vec! [
                        Point::new(2, 0),
                        Point::new(5, 0),
                        Point::new(5, 5),
                        Point::new(3, 1),
                        Point::new(4, 2),
                        Point::new(4, 3),
                        Point::new(4, 4),
                    ]
                },
            ]
        }
    }

    struct FakeRegion {
        points: Vec<Point>,
        neighbours: Vec<Point>,
    }

    impl Region for FakeRegion {
        type NeighbourIter = PointIter;
        type PointIter = PointIter;

        fn neighbours(&self) -> Self::NeighbourIter {
            PointIter { array: self.neighbours.as_ptr(), len: self.neighbours.len(), count: 0 }
        }

        fn points(&self) -> Self::PointIter {
            PointIter { array: self.points.as_ptr(), len: self.points.len(), count: 0 }
        }
    }

    struct FakeAllRegions;

    impl<'a> AllRegions<'a> for FakeAllRegions {
        type Region = FakeRegion;

        fn all(_board: &'a BoardFast, _to_move: Color) -> Vec<Self::Region> {
            vec! [
                FakeRegion {  // a5
                    points: vec! [
                        Point::new(0, 4),
                    ],
                    neighbours: vec! [
                        Point::new(0, 5),
                        Point::new(0, 3),
                        Point::new(1, 4),
                    ],
                },
                FakeRegion {  // b6
                    points: vec! [
                        Point::new(1, 5),
                    ],
                    neighbours: vec! [
                        Point::new(0, 5),
                        Point::new(2, 5),
                        Point::new(1, 4),
                    ],
                },
                FakeRegion {  // c1
                    points: vec! [
                        Point::new(2, 0),
                    ],
                    neighbours: vec! [
                        Point::new(1, 0),
                        Point::new(3, 0),
                        Point::new(2, 1),
                    ],
                },
                FakeRegion {  // f1
                    points: vec! [
                        Point::new(5, 0),
                    ],
                    neighbours: vec! [
                        Point::new(5, 1),
                        Point::new(4, 0),
                    ],
                },
                FakeRegion {  // f6
                    points: vec! [
                        Point::new(5, 5),
                    ],
                    neighbours: vec! [
                        Point::new(5, 4),
                        Point::new(4, 5),
                    ],
                },
                FakeRegion {  // d4
                    points: vec! [
                        Point::new(1, 1),
                        Point::new(1, 2),
                        Point::new(2, 2),
                        Point::new(2, 3),
                        Point::new(3, 1),
                        Point::new(3, 2),
                        Point::new(3, 3),
                        Point::new(3, 4),
                        Point::new(3, 5),
                        Point::new(4, 2),
                        Point::new(4, 3),
                        Point::new(4, 4),
                    ],
                    neighbours: vec! [
                        Point::new(0, 1),
                        Point::new(0, 2),
                        Point::new(1, 0),
                        Point::new(1, 3),
                        Point::new(1, 4),
                        Point::new(2, 1),
                        Point::new(2, 4),
                        Point::new(2, 5),
                        Point::new(3, 0),
                        Point::new(4, 1),
                        Point::new(4, 5),
                        Point::new(5, 2),
                        Point::new(5, 3),
                        Point::new(5, 4),
                    ],
                },
            ]
        }
    }

    /// Example from Sensei Library [1]
    ///
    /// [1] https://senseis.xmp.net/diagrams/7/75bae2500cd8d4922ee8c6659a0666a2.png
    #[test]
    fn benson() {
        let board = Board::new(0.5);
        let benson: Benson<FakeAllRegions, FakeAllBlocks> = Benson::new(&board, Color::Black);
        let invalid = vec! [
            Point::new(0, 0),
            Point::new(0, 1),
            Point::new(0, 2),
            Point::new(0, 3),
            Point::new(0, 4),
            Point::new(0, 5),
            Point::new(1, 0),
            Point::new(1, 3),
            Point::new(1, 4),
            Point::new(1, 5),
            Point::new(2, 4),
            Point::new(2, 5),
        ];

        for &invalid_point in &invalid {
            assert_eq!(benson.is_valid(invalid_point), false);
        }

        for valid_point in Point::all().filter(|p| !invalid.contains(p)) {
            assert_eq!(benson.is_valid(valid_point), true);
        }
    }

    #[test]
    fn empty_is_all_valid() {
        let board = Board::new(0.5);
        let benson = BensonImpl::new(&board, Color::Black);

        for point in Point::all() {
            assert_eq!(benson.is_valid(point), true);
        }
    }

    #[bench]
    fn benson_impl(b: &mut Bencher) {
        let points = vec! [
            (Color::Black, 15, 16), (Color::White, 16,  3), (Color::Black,  3,  2), (Color::White,  3, 15),
            (Color::Black, 14,  3), (Color::White, 14,  2), (Color::Black, 13,  2), (Color::White, 15,  2),
            (Color::Black, 12,  3), (Color::White, 16,  5), (Color::Black,  9,  2), (Color::White,  2,  5),
            (Color::Black,  3,  4), (Color::White,  1,  3), (Color::Black,  3,  5), (Color::White,  2,  6),
            (Color::Black,  3,  6), (Color::White,  2,  7), (Color::Black,  4,  8), (Color::White,  2,  2),
            (Color::Black,  2,  1), (Color::White,  3,  9), (Color::Black,  4,  9), (Color::White,  3, 10),
            (Color::Black,  4, 10), (Color::White,  3, 11), (Color::Black,  2, 16), (Color::White,  2, 15),
            (Color::Black,  3, 16), (Color::White,  5, 16), (Color::Black,  5, 17), (Color::White,  9,  6),
            (Color::Black, 16, 13), (Color::White,  8,  3), (Color::Black,  8,  2), (Color::White,  7,  3),
            (Color::Black,  6, 16), (Color::White,  5, 15), (Color::Black,  7, 17), (Color::White,  9,  3),
            (Color::Black, 10,  2), (Color::White, 13, 16), (Color::Black, 11, 16), (Color::White, 15, 14),
            (Color::Black, 16, 14), (Color::White, 15, 17), (Color::Black, 16, 16), (Color::White, 13, 14),
            (Color::Black, 11, 14), (Color::White, 16, 17), (Color::Black, 17, 17), (Color::White, 14, 16),
            (Color::Black, 16, 18), (Color::White, 14, 18), (Color::Black, 14, 12), (Color::White, 15, 15),
            (Color::Black, 16, 15), (Color::White, 16,  8), (Color::Black, 12, 17), (Color::White, 13, 17),
            (Color::Black, 17, 18), (Color::White, 12, 16), (Color::Black, 11, 17), (Color::White, 11, 13),
            (Color::Black, 10, 13), (Color::White, 11, 12), (Color::Black, 10, 12), (Color::White, 12, 14),
            (Color::Black, 10, 14), (Color::White, 11, 11), (Color::Black, 11,  9), (Color::White, 11,  6),
            (Color::Black,  9,  9), (Color::White, 13,  4), (Color::Black, 12,  4), (Color::White, 13,  5),
            (Color::Black, 10,  5), (Color::White, 10,  6), (Color::Black,  9,  5), (Color::White,  8,  5),
            (Color::Black,  8,  6), (Color::White,  7,  5), (Color::Black,  8,  7), (Color::White,  7,  2),
            (Color::Black, 11,  5), (Color::White, 12,  5), (Color::Black, 10,  3), (Color::White,  4,  1),
            (Color::Black,  4,  2), (Color::White,  5,  2), (Color::Black,  5,  3), (Color::White,  6,  1),
            (Color::Black,  3,  1), (Color::White,  6,  3), (Color::Black,  5,  4), (Color::White, 13,  1),
            (Color::Black, 12,  1), (Color::White, 13,  3), (Color::Black, 12,  2), (Color::White,  8,  1)
        ];
        let mut board = Board::new(0.5);

        for (color, x, y) in points.into_iter() {
            board.place(color, Point::new(x, y));
        }

        b.iter(|| {
            let benson = BensonImpl::new(&board, Color::White);

            for point in Point::all() {
                assert_eq!(benson.is_valid(point), true);
            }
        })
    }

    #[test]
    fn eyes() {
        let mut board = Board::new(0.5);
        board.place(Color::White, Point::new(0, 1));
        board.place(Color::White, Point::new(1, 1));
        board.place(Color::White, Point::new(2, 0));
        board.place(Color::White, Point::new(2, 1));
        board.place(Color::White, Point::new(3, 1));
        board.place(Color::White, Point::new(4, 0));
        board.place(Color::White, Point::new(4, 1));

        board.place(Color::Black, Point::new(0, 0));

        let benson = BensonImpl::new(&board, Color::White);

        for point in Point::all() {
            if board.at(point) == Some(Color::White) {
                assert!(benson.is_alive(point), "{:?} is not alive", point);
            } else if vec! [Point::new(0, 0), Point::new(1, 0), Point::new(3, 0)].contains(&point) {
                assert!(benson.is_eye(point), "{:?} is not eye", point);
            }
        }
    }
}
