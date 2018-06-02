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

use go::{Board, Color, Features, symmetry, CHW};
use mcts::predict::PredictGuard;
use mcts::time_control;
use mcts;
use util::b85;
use util::config;
use util::types::*;

use std::io;

use rand::{self, Rng};
use regex::Regex;

#[derive(Clone)]
enum PolicyEntry {
    Full(String),
    Partial(usize)
}

impl PolicyEntry {
    /// Returns a slice containing the full policy of this entry.
    fn to_slice(&self) -> Vec<f32> {
        match *self {
            PolicyEntry::Full(ref input) => {
                b85::decode::<f16, _>(input).unwrap().into_iter()
                    .map(|value| f32::from(value))
                    .collect()
            },
            PolicyEntry::Partial(index) => {
                let mut policy = vec! [0.0; 362];
                policy[index] = 1.0;
                policy
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
    entries: Vec<(Board, Color, PolicyEntry)>,
    winner: Color,
    server: &'a Option<PredictGuard<'a>>
}

impl<'a> Iterator for EntryIterator<'a> {
    type Item = Entry;

    fn next(&mut self) -> Option<Entry> {
        self.entries.pop()
            .map(|(ref board, current_color, ref policy)| {
                let features = board.get_features::<CHW>(current_color, symmetry::Transform::Identity);
                let policy: Vec<f32> = if self.server.is_some() && policy.is_partial() {
                    // if this is a partial policy then perform a search at this
                    // board position and output the result as the policy
                    let num_threads = ::std::cmp::max(
                        *config::NUM_THREADS / *config::NUM_GAMES,
                        1
                    );
                    let (_, _, tree) = mcts::predict::<mcts::tree::DefaultValue, _>(
                        self.server.as_ref().unwrap(),
                        Some(num_threads),
                        time_control::RolloutLimit::new(*config::NUM_ROLLOUT),
                        None,
                        board,
                        current_color
                    );

                    tree.softmax::<f32>()
                } else {
                    policy.to_slice()
                };

                Entry::new(
                    &features,
                    if current_color == self.winner { 1.0 } else { -1.0 },
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
    /// The current board state, where the last bit represents whether the
    /// current player won the game.
    pub features_and_winner: Vec<u8>,

    /// The probabilities that each move should be played for the given
    /// features, encoded in HW format with one additional element at the
    /// end for the `pass` move.
    pub policy: Vec<u8>
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
    pub fn all<'a>(src: &String, server: &'a Option<PredictGuard>) -> Option<EntryIterator<'a>> {
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

        // instead of plucking `num_samples` examples randomly, just shuffle the
        // list and take the first elements from the array. This avoids picking the
        // same element twice, and automatically handles the case where there are less
        // entries than `num_samples`.
        rand::thread_rng().shuffle(&mut entries);

        Some(EntryIterator {
            entries: entries,
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
    fn new(features: &[i8], winner: f32, policy: &[f32]) -> Entry {
        let mut features = features.to_vec();
        features.push(if winner > 0.0 { 127 } else { 0 });

        Entry {
            features_and_winner: i8_to_bits(&features),
            policy: f32_to_u8(policy)
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
        f.write_all(&self.features_and_winner)?;
        f.write_all(&self.policy)
    }
}

/// Returns an array of bits, where the i:th bit is set if `array[i] = 1.0`.
///
/// # Arguments
///
/// * `array` - the array of 16-bit floating point numbers to serialize
///
fn i8_to_bits(array: &[i8]) -> Vec<u8> {
    let cursor_len = if array.len() % 8 == 0 {
        array.len() / 8
    } else {
        1 + array.len() / 8
    };

    let mut cursor = vec! [0; cursor_len];

    for (i, &value) in array.into_iter().enumerate() {
        if value != 0 {
            let j = i / 8;
            let k = i % 8;

            cursor[j] |= 1 << (7 - k);
        }
    }

    cursor
}

/// Returns an array of floating point serialized (and compressed) as
/// a byte array of FP16.
///
/// # Arguments
///
/// * `array` - the array of 16-bit floating point numbers to serialize
///
fn f32_to_u8(array: &[f32]) -> Vec<u8> {
    let mut cursor = vec! [0; array.len()];

    for (i, &value) in array.into_iter().enumerate() {
        cursor[i] = (255.0 * value).round() as u8;
    }

    cursor
}

#[cfg(test)]
mod tests {
    use dataset::entry::*;

    #[test]
    fn to_bits() {
        let floats = vec! [127, 0, 0, 0, 0, 127, 0, 127];
        let bits = i8_to_bits(&floats);

        assert_eq!(bits.len(), 1);
        assert_eq!(bits[0], 133);
    }

    #[test]
    fn to_u8() {
        let floats = vec! [1.0, 1.0, 0.5, 0.2, 0.7, 0.3, 0.1, 0.8f32];
        let quan = f32_to_u8(&floats);

        assert_eq!(quan.len(), 8);
        assert_eq!(quan.to_vec(), vec! [255, 255, 128, 51, 179, 77, 26, 204]);
    }
}
