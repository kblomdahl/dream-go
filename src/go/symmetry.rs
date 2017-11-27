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

fn get_transformation<F, G>(ax: F, ay: G) -> Box<[usize]>
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

            (19 * ty + tx) as usize
        })
        .collect::<Vec<usize>>()
        .into_boxed_slice()
}

lazy_static! {
    /// Identity transformation.
    static ref _IDENTITY: Box<[usize]> = get_transformation(|x,_| x, |_,y| y);

    /// Flip the matrix across the horizontal axis.
    static ref _FLIP_LR: Box<[usize]> = get_transformation(|x,_| -x, |_,y| y);

    /// Flip the matrix across the vertical axis.
    static ref _FLIP_UD: Box<[usize]> = get_transformation(|x,_| x, |_,y| -y);

    /// Flip the matrix across the main-diagonal.
    static ref _TRANSPOSE_MAIN: Box<[usize]> = get_transformation(|_,y| y, |x,_| x);

    /// Flip the matrix across the anti-diagonal.
    static ref _TRANSPOSE_ANTI: Box<[usize]> = get_transformation(|_,y| -y, |x,_| -x);

    /// Rotate the matrix 90 degrees clock-wise.
    static ref _ROT_90: Box<[usize]> = get_transformation(|_,y| y, |x,_| -x);

    /// Rotate the matrix 180 degrees clock-wise.
    static ref _ROT_180: Box<[usize]> = get_transformation(|x,_| -x, |_,y| -y);

    /// Rotate the matrix 270 degrees clock-wise.
    static ref _ROT_270: Box<[usize]> = get_transformation(|_,y| -y, |x,_| x);
}

/// Available transformations that are part of the go boards symmetry group.
#[derive(Copy, Clone, Debug)]
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
}

/// Helper method that transforms the given values in-place using the
/// given transformation array.
fn apply_aux(values: &mut [f32], transform: &[usize]) {
    assert_eq!(values.len(), 361);

    let mut out = [0.0f32; 361];

    for i in 0..361 {
        let j = transform[i];

        out[j] = values[i];
    }

    values.copy_from_slice(&out);
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
pub fn apply(values: &mut [f32], transform: Transform) {
    let n = values.len() / 361;

    for i in 0..n {
        let s = 361 * i;
        let e = 361 * (i + 1);
        let vs = &mut values[s..e];

        match transform {
            Transform::Identity => apply_aux(vs, &_IDENTITY),
            Transform::FlipLR => apply_aux(vs, &_FLIP_LR),
            Transform::FlipUD => apply_aux(vs, &_FLIP_UD),
            Transform::Transpose => apply_aux(vs, &_TRANSPOSE_MAIN),
            Transform::TransposeAnti => apply_aux(vs, &_TRANSPOSE_ANTI),
            Transform::Rot90 => apply_aux(vs, &_ROT_90),
            Transform::Rot180 => apply_aux(vs, &_ROT_180),
            Transform::Rot270 => apply_aux(vs, &_ROT_270),
        }
    }
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
