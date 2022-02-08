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
#![feature(test)]

extern crate dg_go;
extern crate dg_predict;
extern crate dg_utils;
extern crate test;
extern crate rand;

use test::Bencher;
use rand::{Rng, thread_rng};

use dg_predict::{Builder, Model};
use dg_go::utils::features;
use dg_utils::types::f16;

thread_local! {
    static MODEL: Model = Builder::default().build().expect("could not build predictor model");
}

/// Benchmark the forward pass through the neural network for the given batch
/// size.
///
/// # Arguments
///
/// * `b` -
/// * `batch_size` - the batch size to benchmark
///
fn bench_batch_size(b: &mut Bencher, batch_size: usize) {
    MODEL.with(|model| {
        let features = (0..batch_size).flat_map(|_| {
            let mut input = vec! [f16::from(0.0); features::Default::size()];

            for b in input.iter_mut() {
                *b = f16::from(if thread_rng().gen::<f32>() < 0.2 { 1.0 } else { 0.0 });
            }

            input.into_iter()
        }).collect::<Vec<_>>();

        b.iter(move || {
            let output = unsafe { model.initial_predict(&features, batch_size).unwrap_unchecked() };
            model.predict(&output.hidden_states, &features, batch_size)
        });
    });
}

#[bench] fn batch_size_001(b: &mut Bencher) { bench_batch_size(b,   1); }
#[bench] fn batch_size_002(b: &mut Bencher) { bench_batch_size(b,   2); }
#[bench] fn batch_size_004(b: &mut Bencher) { bench_batch_size(b,   4); }
#[bench] fn batch_size_008(b: &mut Bencher) { bench_batch_size(b,   8); }
#[bench] fn batch_size_016(b: &mut Bencher) { bench_batch_size(b,  16); }
#[bench] fn batch_size_032(b: &mut Bencher) { bench_batch_size(b,  32); }
#[bench] fn batch_size_064(b: &mut Bencher) { bench_batch_size(b,  64); }
#[bench] fn batch_size_128(b: &mut Bencher) { bench_batch_size(b, 128); }
#[bench] fn batch_size_256(b: &mut Bencher) { bench_batch_size(b, 256); }
