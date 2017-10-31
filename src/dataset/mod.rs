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
use std::io::{self, BufReader};

/// The probability that any single position should be included in the
/// data-set.
const KEEP_PROB: f32 = 0.05;

#[derive(Clone)]
pub struct Entry {
    /// The current board state.
    pub features: Box<[f32]>,

    /// The winner for the given features, `1.0` if the current player won
    /// and `-1.0` if the current player lost.
    pub winner: f32,

    /// The probabilities that each move should be played for the given
    /// features, encoded in HW format with one additional element at the
    /// end for the `pass` move.
    pub policy: Box<[f32]>
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
    fn new(src: &String) -> Option<Vec<Entry>> {
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

        let mut entries: Vec<Entry> = vec! [];
        let mut board = Board::new();
        let size = board.size();

        for moves in MOVE.captures_iter(src) {
            let current_color = match &moves[1] {
                "B" => Color::Black,
                "W" => Color::White,
                _   => unreachable!()
            };
            let is_winner = current_color == winner;
            let x = moves[2].chars().nth(0)
                .and_then(|x| LETTERS.binary_search(&x).ok())
                .unwrap_or(board.size());
            let y = moves[2].chars().nth(1)
                .and_then(|y| LETTERS.binary_search(&y).ok())
                .unwrap_or(board.size());

            if x >= board.size() || y >= board.size() {
                if rand::thread_rng().next_f32() < KEEP_PROB {
                    let mut policy = vec! [0.0f32; size*size+1];
                    policy[size*size] = 1.0f32;

                    entries.push(Entry {
                        features: board.get_features(current_color),
                        winner: if is_winner { 1.0 } else { -1.0 },
                        policy: policy.into_boxed_slice()
                    });
                }
            } else if board.is_valid(current_color, x, y) {
                if rand::thread_rng().next_f32() < KEEP_PROB {
                    let mut policy = vec! [0.0f32; size*size+1];
                    policy[size*y + x] = 1.0f32;

                    entries.push(Entry {
                        features: board.get_features(current_color),
                        winner: if is_winner { 1.0 } else { -1.0 },
                        policy: policy.into_boxed_slice()
                    });
                }

                board.place(current_color, x, y);
            } else {
                return None;  // invalid game
            }
        }

        Some(entries)
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
        for &value in self.features.into_iter() {
            write_f32(f, value)?;
        }

        write_f32(f, self.winner)?;

        for &value in self.policy.into_iter() {
            write_f32(f, value)?;
        }

        Ok(())
    }
}

extern "C" {
    #[link_name = "llvm.convert.to.fp16.f32"]
    fn convert_to_fp16_f32(f: f32) -> u16;
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
    let value_b16: u16 = unsafe { convert_to_fp16_f32(value) };

    unsafe {
        let bytes = transmute::<_, [u8; 2]>(value_b16);

        f.write_all(&bytes)
    }
}

/// Iterator over all positions within a single SGF collection, the SGF
/// collection should contain exactly one full game tree per line.
pub struct Dataset {
    /// A buffered reader of the physical SGF file.
    reader: BufReader<File>,

    /// Iterator over the moves of the current line.
    current: Option<::std::vec::IntoIter<Entry>>
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

        Ok(Dataset {
            reader: BufReader::new(handle),
            current: None
        })
    }
}

impl Iterator for Dataset {
    type Item = Entry;

    fn next(&mut self) -> Option<Entry> {
        if self.current.is_some() {
            let entry = self.current.as_mut().and_then(|iter| iter.next());

            if entry.is_some() {
                entry
            } else {
                self.current = None;
                self.next()
            }
        } else {
            let mut line = String::new();
            let result = self.reader.read_line(&mut line);

            if result.is_err() || result.unwrap() == 0 {
                None
            } else {
                self.current = Entry::new(&line).map(|v| v.into_iter());
                self.next()
            }
        }
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
