// Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use std::io::prelude::*;
use std::io::{self, Cursor};

use dataset::crc32c::crc32c_masked;

/* -------- varint -------- */

trait VarInt {
    /// Write this integer, using a variable length encoding, to the given
    /// writer.
    /// 
    /// # Arguments
    /// 
    /// * `out` -
    /// 
    fn write_varint<W: Write>(&self, out: &mut W) -> io::Result<()>;

    /// Returns the number of bytes that will be used to encode this integer.
    fn varint_size(&self) -> usize;
}

impl VarInt for usize {
    fn write_varint<W: Write>(&self, out: &mut W) -> io::Result<()> {
        let mut x = *self;

        while x > 127 {
            out.write_all(&[0x80 | (x & 0x7f) as u8])?;

            x >>= 7;
        }

        out.write_all(&[x as u8])
    }

    fn varint_size(&self) -> usize {
        let mut x = *self;
        let mut n = 1;

        while x > 127 {
            n += 1;
            x >>= 7;
        }

        n
    }
}

/* -------- wire type -------- */

/// _protobuf_ wire types, and their binary representation.
#[repr(u8)]
enum WireType {
    FixedLen = 2
}

impl WireType {
    fn encode(self, field_number: u8) -> u8 {
        (field_number << 3) | (self as u8)
    }
}

/* -------- tfrecord -------- */

/// Write a single protobuf `Feature` with the given name.
/// 
/// # Arguments
/// 
/// * `out` -
/// * `key` -
/// * `value` -
/// 
fn write_bytes_feature<W: Write>(out: &mut W, key: &str, value: &[u8]) -> io::Result<()> {
    let key = key.as_bytes();

    // message Example {
    //   Features features = 1;
    // };
    let bytes_len = 1 + value.len() + value.len().varint_size();
    let feature_len = 1 + bytes_len + bytes_len.varint_size();
    let example_len =
        1 + key.len().varint_size() + key.len() +
        1 + feature_len.varint_size() + feature_len;

    out.write_all(&[WireType::FixedLen.encode(1)])?;
    example_len.write_varint(out)?;

    // message Features {
    //   map<string, Feature> feature = 1;
    // };
    out.write_all(&[WireType::FixedLen.encode(1)])?;
    key.len().write_varint(out)?;
    out.write_all(key)?;

    out.write_all(&[WireType::FixedLen.encode(2)])?;
    feature_len.write_varint(out)?;

    // message Feature {
    //   oneof kind {
    //     BytesList bytes_list = 1;
    //     FloatList float_list = 2;
    //     Int64List int64_list = 3;
    //   }
    // };

    // message BytesList {
    //   repeated bytes value = 1;
    // }
    out.write_all(&[WireType::FixedLen.encode(1)])?;
    bytes_len.write_varint(out)?;

    out.write_all(&[WireType::FixedLen.encode(1)])?;
    value.len().write_varint(out)?;
    out.write_all(value)?;

    Ok(())
}

/// Encode the given `features`, `winner`, and `policy` as a tensorflow
/// `TFRecord` for use during training.
/// 
/// # Arguments
/// 
/// * `features` -
/// * `winner` -
/// * `policy` -
/// 
pub fn encode(features: Vec<u8>, winner: Vec<u8>, policy: Vec<u8>) -> io::Result<Vec<u8>> {
    let mut feature_list = Cursor::new(Vec::with_capacity(512));

    write_bytes_feature(&mut feature_list, "features", &features)?;
    write_bytes_feature(&mut feature_list, "policy", &policy)?;
    write_bytes_feature(&mut feature_list, "value", &winner)?;

    let feature_list = feature_list.into_inner();
    let mut record = Cursor::new(Vec::with_capacity(feature_list.len() + 11));

    record.write_all(&[WireType::FixedLen.encode(1)])?;
    feature_list.len().write_varint(&mut record)?;
    record.write_all(&feature_list)?;

    // A TFRecords file contains a sequence of strings with CRC32C (32-bit CRC
    // using the Castagnoli polynomial) hashes. Each record has the format:
    //
    // uint64 length
    // uint32 masked_crc32_of_length
    // byte   data[length]
    // uint32 masked_crc32_of_data
    let record = record.into_inner();
    let record_len = record.len().to_bytes();
    let mut out = Cursor::new(Vec::with_capacity(record.len() + 16));

    out.write_all(&record_len)?;
    out.write_all(&crc32c_masked(&record_len))?;
    out.write_all(&record)?;
    out.write_all(&crc32c_masked(&record))?;

    Ok(out.into_inner())
}
