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

use ::config;

/// Returns the lower confidence bound of the normal distribution with
/// mean `p_hat`, and `n` samples. Using the confidence interval for `m`
/// visits.
///
/// # Arguments
///
/// * `p_hat` -
/// * `p_std` -
/// * `n` -
/// * `m` -
///
pub fn normal_lcb_m(p_hat: f32, p_std: f32, n: i32, m: i32) -> f32 {
    if n > 0 {
        let z = config::get_lcb_critical_value(m);

        p_hat - z * p_std / (n as f32).sqrt()
    } else {
        0.0
    }
}