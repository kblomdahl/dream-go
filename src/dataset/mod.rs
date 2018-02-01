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

use std::fs::File;
use std::io::{self, BufReader, BufRead};
use std::thread::{self, JoinHandle};
use std::marker::PhantomData;
use std::sync::mpsc::{sync_channel, SyncSender, Receiver};

pub use self::entry::Entry;
use mcts::predict::PredictService;
use util::config;

/// Iterator over all positions within a single SGF collection, the SGF
/// collection should contain exactly one full game tree per line.
pub struct Dataset<'a> {
    /// The channel where finish entries are delivered by the worker
    /// threads.
    receiver: Receiver<Entry>,

    /// The lifetime of the server that each worker thread holds
    lifetime: PhantomData<&'a ()>
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
    pub fn new(src: &str, server: Option<&'a PredictService>) -> Result<Dataset<'a>, io::Error> {
        let handle = File::open(src)?;

        // spawn the worker threads
        let num_games = *config::NUM_GAMES;
        let (t_entry, r_entry) = sync_channel(num_games);
        let workers = (0..num_games).map(|_| {
            let (t_line, r_line) = sync_channel(num_games);
            let t_entry = t_entry.clone();
            let server = server.map(|s| s.lock().clone_static());
            let worker = thread::spawn(move || {
                for line in r_line.iter() {
                    // parse the game, and then send the samples back to the
                    // receivers
                    if let Some(entries) = Entry::all(&line, &server) {
                        let num_samples = ::std::cmp::max(1, match *config::NUM_SAMPLES {
                            config::SamplingStrategy::Percent(pct) => (pct * (entries.original_len() as f32)) as usize,
                            config::SamplingStrategy::Fixed(f) => f
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
pub fn of<'a>(src: &[String], server: Option<&'a PredictService>) -> Datasets<'a> {
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
