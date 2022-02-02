// Copyright 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use crate::options::SearchOptions;
use dg_go::utils::symmetry;
use dg_go::{Point, Board, Color};

/// Returns a initial accumulator policy where all illegal moves has been set
/// to _-Inf_, as well as an symmetry elimination mapping for its indices.
///
/// # Arguments
///
/// * `options` -
/// * `board` -
/// * `color` -
///
pub fn create_initial_policy(
    options: &Box<dyn SearchOptions + Sync>,
    board: &Board,
    to_move: Color
) -> (Vec<f32>, Vec<usize>)
{
    // mark all illegal moves as -Inf, which effectively ensures they are never selected by
    // the tree search.
    let mut policy = vec! [::std::f32::NEG_INFINITY; 368];
    let policy_checker = options.policy_checker(board, to_move);

    for point in Point::all() {
        if policy_checker.is_policy_candidate(board, point) {
            policy[point.to_packed_index()] = 0.0;
        }
    }

    if policy_checker.is_policy_candidate(board, Point::default()) {
        policy[361] = 0.0;
    }

    // remove any symmetric moves that does not contribute to the search.
    //
    // we do this by finding all symmetries which provides symmetric board positions,
    // then for each candidate move we find the minimum index provided by some
    // symmetry.
    let symmetries = symmetry::ALL.iter()
        .filter(|&t| symmetry::is_symmetric(board, *t))
        .collect::<Vec<_>>();
    let mut indices = vec! [0; 362];
    indices[361] = 361;

    for point in Point::all() {
        let i = point.to_packed_index();

        if let Some(target) = symmetries.iter().map(|t| t.apply(point).to_packed_index()).min() {
            indices[i] = target;

            if i != target {
                policy[i] = ::std::f32::NEG_INFINITY;
            }
        } else {
            unreachable!();
        }
    }

    (policy, indices)
}

/// Copy all valid candidates moves from `src` to `dst` applying the given symmetry and
/// the symmetry elimination map.
///
/// # Arguments
///
/// * `dst` -
/// * `src` -
/// * `indices` - the symmetry elimination map
/// * `transform` - the symmetry
///
pub fn add_valid_candidates(
    dst: &mut Vec<f32>,
    src: &[f32],
    indices: &[usize],
    transform: symmetry::Transform
) {
    // always copy the _passing_ move since it is never an illegal move.
    dst[361] += src[361];

    // de-transform each index in the source policy, to the identity board position
    // before adding it to the destination.
    for point in Point::all() {
        let i = point.to_packed_index();
        let j = indices[transform.inverse().apply(point).to_packed_index()];

        dst[j] += src[i];
    }
}

/// Normalize the given vector so that its elements sums to `sum_to`.
///
/// # Arguments
///
/// * `policy` - the vector to normalize in-place
/// * `sum_to` - the value that the elements should sum to
///
pub fn normalize_policy(policy: &mut [f32], sum_to: f32) {
    use crate::asm::sum_finite_f32;
    use crate::asm::normalize_finite_f32;

    // re-normalize the policy since we have modified its values
    let policy_sum: f32 = sum_finite_f32(&policy);

    if policy_sum < 1e-6 {  // do not divide by zero
        let num_finite = policy.iter().filter(|x| x.is_finite()).count() as f32;

        for x in policy.iter_mut().filter(|x| x.is_finite()) {
            *x = sum_to / num_finite;
        }
    } else {
        normalize_finite_f32(policy, policy_sum / sum_to);
    }

    // check for NaN
    for i in 0..362 {
        debug_assert!(!policy[i].is_nan(), "found NaN at index {}, total sum = {}", i, policy_sum);
    }
}
