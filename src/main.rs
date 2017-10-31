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

extern crate dream_go;

use dream_go::dataset;
use std::env;

/// 
fn main() {
    // keep everything that is before the first "--" indicator as potential
    // program arguments
    let args = env::args()
        .skip(1)
        .take_while(|arg| arg != "--")
        .collect::<Vec<String>>();

    // everything after "--" and anything in the potential program arguments
    // that does not begin with a "-"
    let mut remaining = env::args()
        .skip(args.len() + 2)
        .collect::<Vec<String>>();

    for arg in &args {
        if !arg.starts_with("-") {
            remaining.push(arg.clone());
        }
    }

    if args.iter().any(|arg| arg == "--dataset") {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();

        for entry in dataset::of(&remaining) {
            if entry.write_into(&mut handle).is_err() {
                break
            }
        }
    } else if args.iter().any(|arg| arg == "--self-play") {
        unimplemented!();
    } else if args.iter().any(|arg| arg == "--gtp") {
        unimplemented!();
    } else {
        println!("Usage: ./dream-go [options] <files...>");
        println!("");
        println!("  --dataset    Extract a dataset for training from the given SGF files");
        println!("  --self-play  Extract a dataset from self-play");
        println!("  --gtp        Run GTP client");
    }
}
