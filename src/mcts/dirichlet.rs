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

use rand::distributions::{Distribution, Gamma};
use rand::thread_rng;

use util::config;

/// Add a dirichlet distribution of the given scale to `x`.
///
/// # Arguments
///
/// * `x` - the vector to add the distribution to
/// * `scale` - the scale of the distribution
///
pub fn add(x: &mut [f32], shape: f32) {
    add_ex(x, shape, *config::DIRICHLET_NOISE)
}

/// Add a dirichlet distribution of the given scale to `x`.
///
/// # Arguments
///
/// * `x` - the vector to add the distribution to
/// * `scale` - the scale of the distribution
/// * `beta` - the mixing coefficient between the prior value of `x` and
///   the dirichlet distribution.
///
pub fn add_ex(x: &mut [f32], shape: f32, beta: f32) {
    assert!(shape < 1.0);

    let mut g_sum;
    let mut g = vec! [0.0; x.len()];

    loop {
        let gamma = Gamma::new(shape as f64, 1.0);

        g_sum = 0.0;

        for (i, x_) in x.iter().enumerate() {
            if x_.is_finite() {
                let g_ = gamma.sample(&mut thread_rng());

                g_sum += g_;
                g[i] = g_;
            }
        }

        if g_sum > ::std::f64::MIN_POSITIVE {
            break;
        }
    }

    for i in 0..(x.len()) {
        if x[i].is_finite() {
            x[i] = (1.0 - beta) * x[i] + beta * (g[i] / g_sum) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use mcts::dirichlet::*;
    use util::config;

    #[test]
    fn dirichlet() {
        let mut x = vec! [0.0; 1000];
        let mut s = 0.0;
        add(&mut x, 0.03);

        for &v in x.iter() {
            assert!(v.is_finite());
            assert!(v >= 0.0, "{}", v);

            s += v;
        }

        assert!(s >= *config::DIRICHLET_NOISE - 0.01 && s <= *config::DIRICHLET_NOISE + 0.01, "{}", s);
    }

    #[test]
    fn dirichlet_stability() {
        // check if we can re-produce the problem where only a single valid move generate a
        // distribution with a NaN value.
        for _i in 0..10000 {
            let mut x = vec! [0.0; 1];

            add_ex(&mut x, 0.03, 1.0);
            assert_eq!(x[0], 1.0);
        }
    }
}
