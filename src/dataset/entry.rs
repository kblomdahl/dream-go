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

use go::{Board, Color, symmetry, CHW};
use mcts::parallel::Server;
use mcts;
use util::b85;
use util::types::*;

use std::mem::transmute;
use std::io::{self, Cursor};

use rand::{self, Rng};
use regex::Regex;

#[derive(Clone)]
enum PolicyEntry {
    Full(String),
    Partial(usize)
}

impl PolicyEntry {
    /// Returns a slice containing the full policy of this entry.
    fn to_slice(&self) -> Box<[f16]> {
        match *self {
            PolicyEntry::Full(ref input) => b85::decode(input).unwrap(),
            PolicyEntry::Partial(index) => {
                let mut policy = vec! [f16::from(0.0); 362];
                policy[index] = f16::from(1.0);
                policy.into_boxed_slice()
            }
        }
    }

    /// Returns true if this policy does not contain a full policy.
    fn is_partial(&self) -> bool {
        match *self {
            PolicyEntry::Partial(_) => true,
            _ => false
        }
    }
}

pub struct EntryIterator<'a> {
    entries: Vec<((Board, Color, PolicyEntry), &'static symmetry::Transform)>,
    original_size: usize,
    winner: Color,
    server: &'a Option<Server>
}

impl<'a> EntryIterator<'a> {
    /// Returns the number of moves that was played in the game pre-augmentation.
    pub fn original_len(&self) -> usize {
        self.original_size
    }
}

impl<'a> Iterator for EntryIterator<'a> {
    type Item = Entry;

    fn next(&mut self) -> Option<Entry> {
        self.entries.pop()
            .map(|((ref board, current_color, ref policy), &s)| {
                let features = board.get_features::<f16, CHW>(current_color, s);
                let mut policy: Box<[f16]> = if self.server.is_some() && policy.is_partial() {
                    let (_, _, tree) = mcts::predict::<mcts::param::Standard, mcts::tree::DefaultValue>(
                        self.server.as_ref().unwrap(),
                        Some(1),
                        None,
                        board,
                        current_color
                    );

                    tree.softmax::<f16>()
                } else {
                    policy.to_slice()
                };

                symmetry::apply(&mut policy, s);

                Entry::new(
                    &features,
                    f16::from(if current_color == self.winner { 1.0 } else { -1.0 }),
                    &policy
                )
            })
    }
}

impl<'a> ExactSizeIterator for EntryIterator<'a> {
    fn len(&self) -> usize {
        self.entries.len()
    }
}

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
    /// * `server` - the server to use when transforming partial policies to
    ///   full policies. If no server is given then it will emit partial
    ///   policies.
    ///
    pub fn all<'a>(src: &String, server: &'a Option<Server>) -> Option<EntryIterator<'a>> {
        lazy_static! {
            static ref LETTERS: [char; 26] = [
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                'w', 'x', 'y', 'z'
            ];
            static ref WINNER: Regex = Regex::new(r"RE\[([^\]]*)\]").unwrap();
            static ref SCORED: Regex = Regex::new(r"RE\[[BW]\+[0-9\.]+\]").unwrap();
            static ref MOVE: Regex = Regex::new(r";([BW])\[([a-z]*)\](?:P\[([^\]]*)\])?").unwrap();
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

        let mut entries: Vec<(Board, Color, PolicyEntry)> = vec! [];
        let mut board = Board::new();
        let mut pass_count = 0;
        let size = board.size();

        for moves in MOVE.captures_iter(src) {
            let current_color = match &moves[1] {
                "B" => Color::Black,
                "W" => Color::White,
                _   => unreachable!()
            };
            let x = moves[2].chars().nth(0)
                .and_then(|x| LETTERS.binary_search(&x).ok())
                .unwrap_or(size);
            let y = moves[2].chars().nth(1)
                .and_then(|y| LETTERS.binary_search(&y).ok())
                .unwrap_or(size);
            let policy = moves.get(3)
                .map(|input| { PolicyEntry::Full(input.as_str().to_string()) })
                .unwrap_or_else(|| {
                    let index = size*y + x;

                    PolicyEntry::Partial(::std::cmp::min(361, index))
                });

            if x >= size || y >= size {
                entries.push((board.clone(), current_color, policy));
                pass_count += 1;
            } else if board.is_valid(current_color, x, y) {
                entries.push((board.clone(), current_color, policy));
                board.place(current_color, x, y);
                pass_count = 0;
            } else {
                return None;  // invalid game
            }
        }

        // if the game was scored, then add two pass moves at the end of the game
        // since they are missing from a lot of SGF files and we want to engine
        // to learn that one should pass when the game has finished
        if SCORED.is_match(src) && pass_count < 2 {
            let last_color = entries.last().map(|&(_, color, _)| color).unwrap_or(Color::Black);

            if pass_count == 1 && last_color == Color::Black {
                entries.push((board.clone(), Color::White, PolicyEntry::Partial(361)));
            } else if pass_count == 1 && last_color == Color::White {
                entries.push((board.clone(), Color::Black, PolicyEntry::Partial(361)));
            } else {
                entries.push((board.clone(), Color::Black, PolicyEntry::Partial(361)));
                entries.push((board.clone(), Color::White, PolicyEntry::Partial(361)));
            }
        }

        // pluck the specified amount of samples from the game, where each sample
        // is randomly transformed according to one of the symmetries in an attempt
        // to use more samples per file without overfitting
        lazy_static! {
            static ref SYMMETRIES: Vec<symmetry::Transform> = vec! [
                symmetry::Transform::Identity,
                symmetry::Transform::FlipLR,
                symmetry::Transform::FlipUD,
                symmetry::Transform::Transpose,
                symmetry::Transform::TransposeAnti,
                symmetry::Transform::Rot90,
                symmetry::Transform::Rot180,
                symmetry::Transform::Rot270
            ];
        }

        // instead of plucking `num_samples` examples randomly, just shuffle the
        // list and take the first elements from the array. This avoids picking the
        // same element twice, and automatically handles the case where there are less
        // entries than `num_samples`.
        let original_size = entries.len();
        let mut entries: Vec<((Board, Color, PolicyEntry), &symmetry::Transform)> = entries.into_iter()
            .flat_map(|e| ::std::iter::repeat(e).zip(SYMMETRIES.iter()))
            .filter(|&((ref board, _, _), &s)| {
                s == symmetry::Transform::Identity || !symmetry::is_symmetric(board, s)
            })
            .collect();

        rand::thread_rng().shuffle(&mut entries);

        Some(EntryIterator {
            entries: entries,
            original_size: original_size,
            winner: winner,
            server: server
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
    fn new(features: &[f16], winner: f16, policy: &[f16]) -> Entry {
        Entry {
            features: f16_to_bytes(features),
            winner: f16_to_bytes(&[winner]),
            policy: f16_to_bytes(policy)
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
fn write_f16<T>(f: &mut T, value: f16) -> io::Result<()>
    where T: io::Write
{
    unsafe {
        let bytes = transmute::<_, [u8; 2]>(value.to_bits());

        f.write_all(&bytes)
    }
}

/// Returns an array of floating point serialized (and compressed) as
/// a byte array of FP16.
///
/// # Arguments
///
/// * `array` - the array of 16-bit floating point numbers to serialize
///
fn f16_to_bytes(array: &[f16]) -> Box<[u8]> {
    let mut cursor = Cursor::new(vec! [0u8; 2 * array.len()]);

    for &value in array {
        write_f16(&mut cursor, value).unwrap();
    }

    cursor.into_inner().into_boxed_slice()
}
