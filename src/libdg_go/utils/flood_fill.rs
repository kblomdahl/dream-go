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
use crate::point::Point;
use crate::iter::{ChainIter, NextLink};

use std::collections::VecDeque;

pub struct FloodFill {
    head: [Point; Point::MAX],
    next_link: [Point; Point::MAX],
    starting_points: Vec<Point>
}

impl NextLink for &FloodFill {
    fn next_link(&self, point: Point) -> Point {
        self.next_link[point.to_i()]
    }
}

impl FloodFill {
    pub fn new<F, G>(board: &BoardFast, start_at: F, stop_at: G) -> Self
        where F: Fn(&BoardFast, Point) -> bool,
              G: Fn(&BoardFast, Point) -> bool
    {
        let mut out = Self {
            head: [Point::default(); Point::MAX],
            next_link: [Point::default(); Point::MAX],
            starting_points: vec! []
        };

        for point in Point::all() {
            out.head[point] = point;
            out.next_link[point] = point;
        }

        out.flood(board, &start_at, &stop_at);
        out
    }

    fn flood<F, G>(&mut self, board: &BoardFast, start_at: &F, stop_at: &G)
        where F: Fn(&BoardFast, Point) -> bool,
              G: Fn(&BoardFast, Point) -> bool
    {
        for point in Point::all() {
            if self.head[point] == point && start_at(board, point) {
                self.probe(board, point, stop_at);
                self.starting_points.push(point);
            }
        }
    }

    fn probe<F>(&mut self, board: &BoardFast, starting_point: Point, stop_at: &F)
        where F: Fn(&BoardFast, Point) -> bool
    {
        let mut remaining = VecDeque::new();
        remaining.push_back(starting_point);

        while let Some(point) = remaining.pop_front() {
            if self.head[point] == starting_point && point != starting_point {
                continue;
            }

            self.add_to_link(starting_point, point);

            for adj in board.adjacent_to(point) {
                if self.next_link[adj] == adj && !stop_at(board, adj) {
                    remaining.push_back(adj);
                }
            }
        }
    }

    fn add_to_link(&mut self, head: Point, other: Point) {
        debug_assert_eq!(self.head[head], head);

        let head_prev = self.next_link[head];
        let other_prev = self.next_link[other];

        self.head[other] = head;
        self.next_link[head] = other_prev;
        self.next_link[other] = head_prev;
    }

    pub fn starting_points<'a>(&'a self) -> ::std::slice::Iter<'a, Point> {
        self.starting_points.iter()
    }

    pub fn region_at<'a>(&'a self, at: Point) -> ChainIter<&'a FloodFill> {
        ChainIter::new(at, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::color::Color;
    use crate::point_state::Vertex;
    use super::*;

    #[test]
    fn split_down_the_middle() {
        let mut board = Board::new(0.5);
        let blacks = (0..19).map(|i| Point::new(9, i)).collect::<Vec<_>>();

        for point in &blacks {
            board.place(Color::Black, *point);
        }

        let flood =
            FloodFill::new(
                &board.inner,
                |board: &BoardFast, point| board[point].color() == None,
                |board: &BoardFast, point| board[point].color() == Some(Color::Black)
            );

        assert_eq!(flood.starting_points().count(), 2);
    }
}
