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

mod entry;

use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufRead};
use std::thread::{self, JoinHandle};
use std::marker::PhantomData;
use std::sync::mpsc::{sync_channel, SyncSender, Receiver};

pub use self::entry::Entry;
use mcts::parallel::Server;

enum SamplingStrategy {
    Percent(f32),
    Fixed(usize)
}

impl ::std::str::FromStr for SamplingStrategy {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        let s = s.trim();

        if s.ends_with("%") {
            let s = s.trim_right_matches("%");

            s.parse::<f32>()
                .map_err(|_| ())
                .map(|p| SamplingStrategy::Percent(p / 100.0))
        } else {
            s.parse::<usize>()
                .map_err(|_| ())
                .map(|f| SamplingStrategy::Fixed(f))
        }
    }
}

/// Iterator over all positions within a single SGF collection, the SGF
/// collection should contain exactly one full game tree per line.
pub struct Dataset<'a> {
    /// The channel where finish entries are delivered by the worker
    /// threads.
    receiver: Receiver<Entry>,

    /// The lifetime of the server that each worker thread holds
    lifetime: PhantomData<&'a usize>
}

impl<'a> Dataset<'a> {
    /// Returns an iterator over all positions in the given SGF collection.
    ///
    /// # Arguments
    ///
    /// * `src` - the path to the SGF collection
    /// * `server` - the server to use to transform partial policies to
    ///   full policies, if no server is given the partial policies are
    ///   emitted
    ///
    pub fn new(src: &str, server: Option<&'a Server>) -> Result<Dataset<'a>, io::Error> {
        let handle = File::open(src)?;

        // determine what strategy we should use during sampling
        lazy_static! {
            static ref NUM_THREADS: usize = {
                match env::var("NUM_THREADS") {
                    Ok(value) => value.parse::<usize>()
                                      .expect(&format!("NUM_THREADS must be a number, got {}", value)),
                    _ => 32
                }
            };
            static ref STRATEGY: SamplingStrategy = {
                match env::var("NUM_SAMPLES") {
                    Ok(value) => {
                        value.parse::<SamplingStrategy>().ok()
                            .expect(&format!("NUM_SAMPLES must be a number or a percentage, e.g. \"3\" or \"5%\", got {}", value))
                    },
                    _ => SamplingStrategy::Fixed(1)
                }
            };
        }

        // spawn the worker threads
        let (t_entry, r_entry) = sync_channel(*NUM_THREADS);
        let workers = (0..*NUM_THREADS).map(|_| {
            let (t_line, r_line) = sync_channel(*NUM_THREADS);
            let t_entry = t_entry.clone();
            let server = server.map(|&ref server| server.clone());
            let worker = thread::spawn(move || {
                for line in r_line.iter() {
                    if let Some(entries) = Entry::all(&line, &server) {
                        let num_samples = ::std::cmp::max(1, match *STRATEGY {
                            SamplingStrategy::Percent(pct) => (pct * (entries.original_len() as f32)) as usize,
                            SamplingStrategy::Fixed(f) => f
                        });

                        for entry in entries.take(num_samples) {
                            t_entry.send(entry).unwrap();
                        }
                    }
                }

                drop(t_entry);
            });

            (worker, t_line)
        }).collect::<Vec<(JoinHandle<()>, SyncSender<String>)>>();

        // spawn the thread that is responsible for distributing the work
        // over to all of the worker threads
        thread::spawn(move || {
            let reader = BufReader::new(handle);

            for (i, result) in reader.lines().enumerate() {
                if let Ok(line) = result {
                    let tx = &workers[i % workers.len()].1;

                    tx.send(line).unwrap();
                }
            }

            // terminate all worker threads
            for (worker, tx) in workers.into_iter() {
                drop(tx);
                worker.join().unwrap();
            }
        });

        Ok(Dataset {
            receiver: r_entry,
            lifetime: PhantomData
        })
    }
}

impl<'a> Iterator for Dataset<'a> {
    type Item = Entry;

    fn next(&mut self) -> Option<Entry> {
        self.receiver.recv().ok()
    }
}

pub struct Datasets<'a> {
    /// The sets to chain over
    sets: Vec<Dataset<'a>>,

    /// The index of the current set we are iterating over
    current: usize
}

impl<'a> Iterator for Datasets<'a> {
    type Item = Entry;

    fn next(&mut self) -> Option<Entry> {
        if self.current < self.sets.len() {
            let entry = self.sets[self.current].next();

            if entry.is_some() {
                entry
            } else {
                self.current += 1;
                self.next()
            }
        } else {
            None
        }
    }
}

/// Returns an iterator over all positions in the given SGF files.
///
/// # Arguments
///
/// * `src` - the path to the SGF files
/// * `server` - the server to use to transform partial policies to
///   full policies, if no server is given the partial policies are
///   emitted
///
pub fn of<'a>(src: &[String], server: Option<&'a Server>) -> Datasets<'a> {
    Datasets {
        sets: src.iter()
            .filter_map(|f| {
                let result = Dataset::new(f, server);

                result.ok()
            })
            .collect::<Vec<Dataset>>(),
        current: 0
    }
}
