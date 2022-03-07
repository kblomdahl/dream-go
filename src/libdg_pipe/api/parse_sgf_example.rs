// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use super::{Example, Game};
use dg_go::utils::symmetry;

use libc::{c_int, size_t};
use rand::prelude::*;

#[no_mangle]
pub extern "C" fn parse_sgf_example(
    example: *mut Example,
    content: *const u8,
    content_length: size_t,
) -> c_int
{
    if content.is_null() || example.is_null() {
        return -14; // EFAULT
    }

    let content = unsafe { ::std::slice::from_raw_parts(content, content_length) };
    let example = unsafe { &mut (*example) };

    if !example.has_shape() || !example.has_ptr() {
        return -22; // EINVAL
    }

    if let Some(game) = Game::from_bytes(content) {
        parse_game_examples(example, &game)
    } else {
        -84 // EILSEQ
    }
}

fn parse_game_examples(example: &mut Example, game: &Game) -> c_int {
    let num_examples = example.features_shape[0] as usize;
    let num_unrolls = example.features_shape[1] as usize;

    if game.len() < (num_examples * num_unrolls) {
        return -101; // game is too short
    }

    let mut free_indices = (0..game.len() - num_unrolls).collect::<Vec<_>>();

    for i in 0..num_examples {
        if let Some(starting_index) = free_indices.choose(&mut thread_rng()).cloned() {
            let symmetry = symmetry::ALL.choose(&mut thread_rng()).cloned().unwrap();

            for j in 0..num_unrolls {
                let idx = starting_index + j;
                let cand = &game[idx];

                example.features(i, j).copy_from_slice(&cand.features(symmetry));
                example.motion_features(i, j).copy_from_slice(&cand.motion_features(symmetry));
                example.additional_targets(i, j).copy_from_slice(&cand.additional_targets(symmetry));
                example.additional_targets_mask(i, j).copy_from_slice(&cand.additional_targets_mask());
                example.lz_features(i, j).copy_from_slice(&cand.lz_features(&game, idx, symmetry));
                example.value(i, j).copy_from_slice(&[cand.value()]);
                example.policy(i, j).copy_from_slice(&cand.policy(symmetry));
            }

            // even after removing these indices we can still get overlapping
            // regions, but that's fine.
            free_indices.retain(|idx| !(starting_index..(starting_index + num_unrolls)).contains(idx));
        } else {
            return -101; // game is too short
        }
    }

    0
}
