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
//
#![feature(test)]

extern crate dream_go;
extern crate test;
extern crate rand;

use test::Bencher;
use rand::{Rng, thread_rng};

use dream_go::nn::*;
use dream_go::util::f16::*;

thread_local! {
    static NETWORK: Network = Network::new().unwrap();
}

/// Benchmark the forward pass through the neural network for the given batch
/// size and data type.
/// 
/// # Arguments
/// 
/// * `b` -
/// * `network` -
/// * `batch_size` - the batch size to benchmark
/// 
fn bench_batch_size_aux<T>(
    b: &mut Bencher,
    network: &Network,
    batch_size: usize
)
    where T: From<f32> + Clone
{
    let mut workspace = network.get_workspace(batch_size);
    let features = (0..batch_size).map(|_| {
        let mut input = vec! [T::from(0.0); 11552];

        for b in input.iter_mut() {
            *b = T::from(if thread_rng().next_f32() < 0.2 { 1.0 } else { 0.0 });
        }

        input.into_boxed_slice()
    }).collect();

    b.iter(move || {
        forward(&mut workspace, &features)
    });
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
    NETWORK.with(|network| {
        // allocate a feature vector filled with random ones and zeros
        if network.is_half() {
            bench_batch_size_aux::<f16>(b, network, batch_size);
        } else {
            bench_batch_size_aux::<f32>(b, network, batch_size);
        }
    });
}

#[bench] fn batch_size_01(b: &mut Bencher) { bench_batch_size(b,  1); }
#[bench] fn batch_size_02(b: &mut Bencher) { bench_batch_size(b,  2); }
#[bench] fn batch_size_04(b: &mut Bencher) { bench_batch_size(b,  4); }
#[bench] fn batch_size_08(b: &mut Bencher) { bench_batch_size(b,  8); }
#[bench] fn batch_size_16(b: &mut Bencher) { bench_batch_size(b, 16); }
#[bench] fn batch_size_32(b: &mut Bencher) { bench_batch_size(b, 32); }
#[bench] fn batch_size_64(b: &mut Bencher) { bench_batch_size(b, 64); }
