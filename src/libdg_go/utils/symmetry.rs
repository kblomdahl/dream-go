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

use board_fast::{Vertex};
use board::Board;
use point::Point;

fn get_transformation<F, G>(ax: F, ay: G) -> Box<[Point]>
    where F: Fn(i32, i32) -> i32, G: Fn(i32, i32) -> i32
{
    let mut out = vec! [Point::default(); Point::MAX];

    for point in Point::all() {
        let x = point.x() as i32;
        let y = point.y() as i32;
        let tx = ax(x - 9, y - 9) + 9;
        let ty = ay(x - 9, y - 9) + 9;

        assert!(tx >= 0 && tx < 19, "tx {} -> {}", x, tx);
        assert!(ty >= 0 && ty < 19, "ty {} -> {}", y, ty);

        out[point.to_i()] = Point::new(tx as usize, ty as usize);
    }

    out.into_boxed_slice()
}

lazy_static! {
    /// Identity transformation.
    static ref _IDENTITY: Box<[Point]> = get_transformation(|x,_| x, |_,y| y);

    /// Flip the matrix across the horizontal axis.
    static ref _FLIP_LR: Box<[Point]> = get_transformation(|x,_| -x, |_,y| y);

    /// Flip the matrix across the vertical axis.
    static ref _FLIP_UD: Box<[Point]> = get_transformation(|x,_| x, |_,y| -y);

    /// Flip the matrix across the main-diagonal.
    static ref _TRANSPOSE_MAIN: Box<[Point]> = get_transformation(|_,y| y, |x,_| x);

    /// Flip the matrix across the anti-diagonal.
    static ref _TRANSPOSE_ANTI: Box<[Point]> = get_transformation(|_,y| -y, |x,_| -x);

    /// Rotate the matrix 90 degrees clock-wise.
    static ref _ROT_90: Box<[Point]> = get_transformation(|_,y| y, |x,_| -x);

    /// Rotate the matrix 180 degrees clock-wise.
    static ref _ROT_180: Box<[Point]> = get_transformation(|x,_| -x, |_,y| -y);

    /// Rotate the matrix 270 degrees clock-wise.
    static ref _ROT_270: Box<[Point]> = get_transformation(|_,y| -y, |x,_| x);
}

/// Available transformations that are part of the go boards symmetry group.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Transform {
    Identity,
    FlipLR,
    FlipUD,
    Transpose,
    TransposeAnti,
    Rot90,
    Rot180,
    Rot270
}

impl Transform {
    pub fn inverse(self) -> Transform {
        match self {
            Transform::Identity => Transform::Identity,
            Transform::FlipLR => Transform::FlipLR,
            Transform::FlipUD => Transform::FlipUD,
            Transform::Transpose => Transform::Transpose,
            Transform::TransposeAnti => Transform::TransposeAnti,
            Transform::Rot90 => Transform::Rot270,
            Transform::Rot180 => Transform::Rot180,
            Transform::Rot270 => Transform::Rot90
        }
    }

    pub fn apply(self, index: Point) -> Point {
        match self {
            Transform::Identity => _IDENTITY[index],
            Transform::FlipLR => _FLIP_LR[index],
            Transform::FlipUD => _FLIP_UD[index],
            Transform::Transpose => _TRANSPOSE_MAIN[index],
            Transform::TransposeAnti => _TRANSPOSE_ANTI[index],
            Transform::Rot90 => _ROT_90[index],
            Transform::Rot180 => _ROT_180[index],
            Transform::Rot270 => _ROT_270[index],
        }
    }

    pub fn get_table(self) -> &'static [Point] {
        match self {
            Transform::Identity => &_IDENTITY,
            Transform::FlipLR => &_FLIP_LR,
            Transform::FlipUD => &_FLIP_UD,
            Transform::Transpose => &_TRANSPOSE_MAIN,
            Transform::TransposeAnti => &_TRANSPOSE_ANTI,
            Transform::Rot90 => &_ROT_90,
            Transform::Rot180 => &_ROT_180,
            Transform::Rot270 => &_ROT_270
        }
    }
}

pub static ALL: [Transform; 8] = [
    Transform::Identity,
    Transform::FlipLR,
    Transform::FlipUD,
    Transform::Transpose,
    Transform::TransposeAnti,
    Transform::Rot90,
    Transform::Rot180,
    Transform::Rot270
];

/// Returns if the given board is symmetric over the given group.
///
/// # Arguments
///
/// * `board` -
/// * `transform` -
///
pub fn is_symmetric(board: &Board, transform: Transform) -> bool {
    let lookup: &[Point] = transform.get_table();

    Point::all().all(|i| {
        let j = lookup[i];

        board.inner[i].color() == board.inner[j].color()
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use super::*;

    fn test_symmetry(t: Transform) {
        let mut seen = HashSet::new();

        for point in Point::all() {
            let other = t.apply(point);

            assert!(seen.insert(other));
        }
    }

    #[test]
    pub fn identity() {
        test_symmetry(Transform::Identity);
    }

    #[test]
    pub fn flip_lr() {
        test_symmetry(Transform::FlipLR);
    }

    #[test]
    pub fn flip_ud() {
        test_symmetry(Transform::FlipUD);
    }

    #[test]
    pub fn transpose() {
        test_symmetry(Transform::Transpose);
    }

    #[test]
    pub fn transpose_anti() {
        test_symmetry(Transform::TransposeAnti);
    }

    #[test]
    pub fn rot90() {
        test_symmetry(Transform::Rot90);
    }

    #[test]
    pub fn rot180() {
        test_symmetry(Transform::Rot180);
    }

    #[test]
    pub fn rot270() {
        test_symmetry(Transform::Rot270);
    }
}
