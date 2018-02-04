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

use std::cell::RefCell;
use std::mem;

use go::Board;

fn get_transformation<F, G>(ax: F, ay: G) -> Box<[u16]>
    where F: Fn(i32, i32) -> i32, G: Fn(i32, i32) -> i32
{
    (0..361)
        .map(|i| {
            let x = (i % 19) as i32;
            let y = (i / 19) as i32;
            let tx = ax(x - 9, y - 9) + 9;
            let ty = ay(x - 9, y - 9) + 9;

            assert!(tx >= 0 && tx < 19, "tx {} -> {}", x, tx);
            assert!(ty >= 0 && ty < 19, "ty {} -> {}", y, ty);

            (19 * ty + tx) as u16
        })
        .collect::<Vec<u16>>()
        .into_boxed_slice()
}

lazy_static! {
    /// Identity transformation.
    static ref _IDENTITY: Box<[u16]> = get_transformation(|x,_| x, |_,y| y);

    /// Flip the matrix across the horizontal axis.
    static ref _FLIP_LR: Box<[u16]> = get_transformation(|x,_| -x, |_,y| y);

    /// Flip the matrix across the vertical axis.
    static ref _FLIP_UD: Box<[u16]> = get_transformation(|x,_| x, |_,y| -y);

    /// Flip the matrix across the main-diagonal.
    static ref _TRANSPOSE_MAIN: Box<[u16]> = get_transformation(|_,y| y, |x,_| x);

    /// Flip the matrix across the anti-diagonal.
    static ref _TRANSPOSE_ANTI: Box<[u16]> = get_transformation(|_,y| -y, |x,_| -x);

    /// Rotate the matrix 90 degrees clock-wise.
    static ref _ROT_90: Box<[u16]> = get_transformation(|_,y| y, |x,_| -x);

    /// Rotate the matrix 180 degrees clock-wise.
    static ref _ROT_180: Box<[u16]> = get_transformation(|x,_| -x, |_,y| -y);

    /// Rotate the matrix 270 degrees clock-wise.
    static ref _ROT_270: Box<[u16]> = get_transformation(|_,y| -y, |x,_| x);
}

/// Available transformations that are part of the go boards symmetry group.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
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
    pub fn inverse(&self) -> Transform {
        match *self {
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

    pub fn apply(&self, index: usize) -> usize {
        let dest = match *self {
            Transform::Identity => _IDENTITY[index],
            Transform::FlipLR => _FLIP_LR[index],
            Transform::FlipUD => _FLIP_UD[index],
            Transform::Transpose => _TRANSPOSE_MAIN[index],
            Transform::TransposeAnti => _TRANSPOSE_ANTI[index],
            Transform::Rot90 => _ROT_90[index],
            Transform::Rot180 => _ROT_180[index],
            Transform::Rot270 => _ROT_270[index],
        };

        dest as usize
    }

    pub fn get_table(&self) -> &'static [u16] {
        match *self {
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

fn reorder<T: Copy>(src: &[T], dst: &mut [T], mapping: &[u16]) {
    unsafe {
        for i in 0..361 {
            let j = *mapping.get_unchecked(i) as usize;

            *dst.get_unchecked_mut(j) = *src.get_unchecked(i);
        }
    }
}

/// Apply the given symmetry transformation to the tensor in CHW
/// format, with any left-over elements at the end of the tensor
/// untouched. The transformation is applied in-place.
/// 
/// # Arguments
/// 
/// * `values` -
/// * `transform` - 
/// 
pub fn apply<T: Copy>(values: &mut [T], transform: Transform) {
    thread_local! {
        static WORKSPACE: RefCell<[i32; 361]> = RefCell::new([0; 361]);
    }

    debug_assert!(mem::size_of::<T>() <= 4);

    WORKSPACE.with(|workspace| { unsafe {
        let mut workspace = workspace.borrow_mut();
        let workspace = &mut *(&mut *workspace as *mut [i32] as *mut [T]);
        let lookup: &[u16] = transform.get_table();
        let n = values.len() / 361;

        for i in 0..n {
            let s = 361 * i;
            let e = 361 * (i + 1);

            reorder(
                &values[s..e],
                workspace,
                lookup
            );

            values[s..e].copy_from_slice(workspace);
        }
    }});
}

/// Returns if the given board is symmetric over the given group.
/// 
/// # Arguments
/// 
/// * `board` -
/// * `transform` -
/// 
pub fn is_symmetric(board: &Board, transform: Transform) -> bool {
    let lookup: &[u16] = transform.get_table();

    (0..361).all(|i| {
        let j = lookup[i] as usize;

        board.inner.vertices[i] == board.inner.vertices[j]
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use go::*;

    fn test_uniq(values: &[f32]) {
        let u = values.into_iter()
            .map(|&x| x as i32)
            .inspect(|&x| {
                assert!(x >= 0 && x < 361);
            })
            .collect::<HashSet<i32>>();

        assert_eq!(u.len(), 361);
    }

    #[test]
    pub fn identity() {
        let mut seq = (0..361).map(|i| i as f32).collect::<Vec<f32>>();
        symmetry::apply(&mut seq, symmetry::Transform::Identity);

        test_uniq(&seq);
    }

    #[test]
    pub fn flip_lr() {
        let mut seq = (0..361).map(|i| i as f32).collect::<Vec<f32>>();
        symmetry::apply(&mut seq, symmetry::Transform::FlipLR);

        test_uniq(&seq);
    }

    #[test]
    pub fn flip_ud() {
        let mut seq = (0..361).map(|i| i as f32).collect::<Vec<f32>>();
        symmetry::apply(&mut seq, symmetry::Transform::FlipUD);

        test_uniq(&seq);
    }

    #[test]
    pub fn transpose() {
        let mut seq = (0..361).map(|i| i as f32).collect::<Vec<f32>>();
        symmetry::apply(&mut seq, symmetry::Transform::Transpose);

        test_uniq(&seq);
    }

    #[test]
    pub fn transpose_anti() {
        let mut seq = (0..361).map(|i| i as f32).collect::<Vec<f32>>();
        symmetry::apply(&mut seq, symmetry::Transform::TransposeAnti);

        test_uniq(&seq);
    }

    #[test]
    pub fn rot90() {
        let mut seq = (0..361).map(|i| i as f32).collect::<Vec<f32>>();
        symmetry::apply(&mut seq, symmetry::Transform::Rot90);

        test_uniq(&seq);
    }

    #[test]
    pub fn rot180() {
        let mut seq = (0..361).map(|i| i as f32).collect::<Vec<f32>>();
        symmetry::apply(&mut seq, symmetry::Transform::Rot180);

        test_uniq(&seq);
    }

    #[test]
    pub fn rot270() {
        let mut seq = (0..361).map(|i| i as f32).collect::<Vec<f32>>();
        symmetry::apply(&mut seq, symmetry::Transform::Rot270);

        test_uniq(&seq);
    }
}
