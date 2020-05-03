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


/// Returns the weighted n:th percentile of the given array, and the sum of
/// all smaller elements.
///
/// # Arguments
///
/// * `array` -
/// * `total` -
/// * `n` - the percentile to get
///
fn percentile<O: Ord + Clone + Copy + Default + Into<f64>>(
    items: &[O],
    total: f64,
    n: f64
) -> Option<(O, f64)>
{
    debug_assert!(n >= 0.0 && n <= 1.0);

    let mut indices = (0..items.len()).collect::<Vec<_>>();
    indices.sort_unstable_by_key(|&i| items[i]);

    // step forward in the array until we have accumulated the requested amount
    let max_value = total * (1.0 - n);
    let mut so_far = 0.0;

    for i in indices.into_iter().rev() {
        so_far += items[i].into();

        if so_far >= max_value {
            return Some((items[i], so_far));
        }
    }

    None
}

/// Choose the smallest value in `items` whose cumulative sum is at least `at`
/// percent of the total cumulative value. This is done only on the
/// `cutoff_percentile` smallest values has been removed.
/// 
/// # Arguments
/// 
/// * `items` -
/// * `cutoff_percentile` - the percent of the cumulative value to prune
/// * `temperature` -
/// * `at` -
/// 
pub fn choose<O: Ord + Clone + Copy + Default + Into<f64>>(
    items: &[O],
    cutoff_percentile: f64,
    temperature: f64,
    at: f64
) -> Option<(usize, f64)>
{
    debug_assert!(at >= 0.0 && at <= 1.0);

    let total = items.iter()
        .map(|&x| x.into())
        .filter(|&x| x.is_finite())
        .sum::<f64>();
    let (threshold, total) = percentile(
        items,
        total,
        cutoff_percentile
    )?;

    let mut cum = vec! [::std::f64::NAN; items.len()];
    let mut cum_total = 0.0;

    for (i, &x) in items.iter().enumerate() {
        if x >= threshold {
            cum_total += (x.into() / total).powf(temperature);
            cum[i] = cum_total;
        }
    }

    // adjust `at` since the temperature might have changed the sum
    let at = at * cum_total;

    cum.into_iter().enumerate().find(|(_, x)| *x >= at)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_50() {
        let (val, total) = percentile(
            &mut vec! [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            45.0,
            0.5
        ).unwrap();

        assert_eq!(val, 7);
        assert_eq!(total, 24.0);
    }

    #[test]
    fn choose_1() {
        let expected = Some((2, 4.0 / 7.0));

        assert_eq!(
            choose(
                &mut vec! [2, 1, 4, 3],
                0.5,
                1.0,
                0.0
            ),
            expected
        );
    }

    #[test]
    fn choose_2() {
        let expected = Some((1, 1.0));

        assert_eq!(
            choose(
                &mut vec! [4, 3, 2, 1],
                0.5,
                1.0,
                1.0
            ),
            expected
        );
    }
}
