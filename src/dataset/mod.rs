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

use go::{Board, Color};
use regex::Regex;

use rand::{self, Rng};
use std::fs::File;
use std::mem::transmute;
use std::io::prelude::*;
use std::io::{self, BufReader, Cursor};
use std::thread::{self, JoinHandle};
use std::sync::mpsc::{sync_channel, SyncSender, Receiver};
use ::f16::*;

#[derive(Clone)]
pub struct Entry {
    /// The current board state.
    pub features: Box<[u8]>,

    /// The winner for the given features, `1.0` if the current player won
    /// and `-1.0` if the current player lost.
    pub winner: Box<[u8]>,

    /// The probabilities that each move should be played for the given
    /// features, encoded in HW format with one additional element at the
    /// end for the `pass` move.
    pub policy: Box<[u8]>
}

impl Entry {
    /// Returns all entries that can be extracted from the SGF file contained
    /// in the given string. If the given game contains invalid moves, or does
    /// not have a recorded winner then `None` is returned.
    ///
    /// # Arguments
    ///
    /// * `src` - the SGF game
    ///
    fn all(src: &String) -> Option<Vec<Entry>> {
        lazy_static! {
            static ref LETTERS: [char; 26] = [
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                'w', 'x', 'y', 'z'
            ];
            static ref WINNER: Regex = Regex::new(r"RE\[([^\]]*)\]").unwrap();
            static ref MOVE: Regex = Regex::new(r";([BW])\[([a-z]*)\]").unwrap();
        }

        let winner = {
            if let Some(caps) = WINNER.captures(src) {
                match caps[1].chars().nth(0) {
                    Some('B') => Color::Black,
                    Some('W') => Color::White,
                    _   => { return None; }
                }
            } else {
                return None
            }
        };

        let mut entries: Vec<(Board, Color, usize)> = vec! [];
        let mut board = Board::new();
        let size = board.size();

        for moves in MOVE.captures_iter(src) {
            let current_color = match &moves[1] {
                "B" => Color::Black,
                "W" => Color::White,
                _   => unreachable!()
            };
            let x = moves[2].chars().nth(0)
                .and_then(|x| LETTERS.binary_search(&x).ok())
                .unwrap_or(board.size());
            let y = moves[2].chars().nth(1)
                .and_then(|y| LETTERS.binary_search(&y).ok())
                .unwrap_or(board.size());

            if x >= board.size() || y >= board.size() {
                entries.push((board.clone(), current_color, size*size));
            } else if board.is_valid(current_color, x, y) {
                entries.push((board.clone(), current_color, size*y + x));
                board.place(current_color, x, y);
            } else {
                return None;  // invalid game
            }
        }

        // pick exactly one entry from each game
        rand::thread_rng().choose(&entries)
                .map(|&(ref board, current_color, correct_index)| {
                    let mut policy = vec! [0.0f32; 362];
                    policy[correct_index] = 1.0;

                    vec! [Entry::new(
                        &board.get_features(current_color),
                        if current_color == winner { 1.0 } else { -1.0 },
                        &policy
                    )]
                })
    }

    /// Returns an entry with the given values stored as compressed FP16
    /// buffers.
    ///
    /// # Arguments
    ///
    /// * `features` - the features of the board state
    /// * `winner` - the winner
    /// * `policy` - the policy vector
    ///
    fn new(features: &[f32], winner: f32, policy: &[f32]) -> Entry {
        Entry {
            features: f32_to_f16(features),
            winner: f32_to_f16(&[winner]),
            policy: f32_to_f16(policy)
        }
    }

    /// Write a binary representation of this entry to the given formatter.
    ///
    /// # Arguments
    ///
    /// * `f` - the formatter to write this entry to
    ///
    pub fn write_into<T>(&self, f: &mut T) -> io::Result<()>
        where T: io::Write
    {
        f.write_all(&self.features)?;
        f.write_all(&self.winner)?;
        f.write_all(&self.policy)
    }
}

/// Write the given 32-bit floating point number to the given formatter
/// in the platform endianess.
///
/// # Arguments
///
/// * `f` - the formatter to write to
/// * `value` - the value to write
///
fn write_f32<T>(f: &mut T, value: f32) -> io::Result<()>
    where T: io::Write
{
    let value_b16: u16 = f16::from(value).to_bits();

    unsafe {
        let bytes = transmute::<_, [u8; 2]>(value_b16);

        f.write_all(&bytes)
    }
}

/// Returns an array of floating point serialized (and compressed) as FP16.
///
/// # Arguments
///
/// * `array` - the array of 32-bit floating point numbers to compress
///
fn f32_to_f16(array: &[f32]) -> Box<[u8]> {
    let mut cursor = Cursor::new(vec! [0u8; 2 * array.len()]);

    for &value in array {
        write_f32(&mut cursor, value).unwrap();
    }

    cursor.into_inner().into_boxed_slice()
}

/// Iterator over all positions within a single SGF collection, the SGF
/// collection should contain exactly one full game tree per line.
pub struct Dataset {
    /// The channel where finish entries are delivered by the worker
    /// threads.
    receiver: Receiver<Entry>
}

impl Dataset {
    /// Returns an iterator over all positions in the given SGF collection.
    ///
    /// # Arguments
    ///
    /// * `src` - the path to the SGF collection
    ///
    pub fn new(src: &str) -> Result<Dataset, io::Error> {
        let handle = File::open(src)?;

        // spawn the worker threads
        const NUM_THREADS: usize = 32;

        let (t_entry, r_entry) = sync_channel(NUM_THREADS);
        let workers = (0..NUM_THREADS).map(|_| {
            let (t_line, r_line) = sync_channel(NUM_THREADS);
            let t_entry = t_entry.clone();
            let worker = thread::spawn(move || {
                for line in r_line.iter() {
                    if let Some(entries) = Entry::all(&line) {
                        for entry in entries {
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
            receiver: r_entry
        })
    }
}

impl Iterator for Dataset {
    type Item = Entry;

    fn next(&mut self) -> Option<Entry> {
        self.receiver.recv().ok()
    }
}

pub struct Datasets {
    /// The sets to chain over
    sets: Vec<Dataset>,

    /// The index of the current set we are iterating over
    current: usize
}

impl Iterator for Datasets {
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
///
pub fn of(src: &[String]) -> Datasets {
    Datasets {
        sets: src.iter()
            .filter_map(|f| {
                let result = Dataset::new(f);

                result.ok()
            })
            .collect::<Vec<Dataset>>(),
        current: 0
    }
}
