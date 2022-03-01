// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use dg_go::{Color, Point, Board};

pub trait SgfFormat {
    fn x_as_str(x: usize) -> u8 {
        b'a' + x as u8
    }

    fn y_as_str(y: usize) -> u8 {
        b'a' + y as u8
    }
}

pub struct CGoban;
pub struct Sabaki;

impl SgfFormat for CGoban {}
impl SgfFormat for Sabaki {
    fn y_as_str(y: usize) -> u8 {
        b'a' + (18 - y) as u8
    }
}

pub trait ToSgf {
    fn to_sgf<F: SgfFormat>(&self) -> String;
}

impl ToSgf for Color {
    fn to_sgf<F: SgfFormat>(&self) -> String {
        match self {
            Self::Black => "B".to_string(),
            Self::White => "W".to_string(),
        }
    }
}

impl ToSgf for Point {
    fn to_sgf<F: SgfFormat>(&self) -> String {
        if *self == Point::default() {
            "".to_string()
        } else {
            unsafe {
                String::from_utf8_unchecked(vec! [
                    F::x_as_str(self.x()),
                    F::y_as_str(self.y()),
                ])
            }
        }
    }
}

impl ToSgf for Board {
    fn to_sgf<F: SgfFormat>(&self) -> String {
        let black = Point::all()
            .filter_map(|v| if self.at(v) == Some(Color::Black) { Some(v.to_sgf::<F>()) } else { None })
            .collect::<Vec<_>>();
        let white = Point::all()
            .filter_map(|v| if self.at(v) == Some(Color::White) { Some(v.to_sgf::<F>()) } else { None })
            .collect::<Vec<_>>();

        if black.is_empty() && white.is_empty() {
            format!("(;)")
        } else if black.is_empty() {
            format!("(;AW[{}])", white.join("]["))
        } else if white.is_empty() {
            format!("(;AB[{}])", black.join("]["))
        } else {
            format!("(;AB[{}]AW[{}])", black.join("]["), white.join("]["))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black() {
        assert_eq!(Color::Black.to_sgf::<CGoban>(), "B");
    }

    #[test]
    fn white() {
        assert_eq!(Color::White.to_sgf::<CGoban>(), "W");
    }

    #[test]
    fn d4() {
        assert_eq!(Point::new(3, 3).to_sgf::<CGoban>(), "dd");
    }

    #[test]
    fn d16() {
        assert_eq!(Point::new(3, 15).to_sgf::<CGoban>(), "dp");
    }

    #[test]
    fn k10() {
        assert_eq!(Point::new(9, 9).to_sgf::<CGoban>(), "jj");
    }

    #[test]
    fn pass() {
        assert_eq!(Point::default().to_sgf::<CGoban>(), "");
    }

    #[test]
    fn board_d4() {
        let mut board = Board::new(0.5);
        board.place(Color::Black, Point::new(3, 3));

        assert_eq!(board.to_sgf::<CGoban>(), "(;AB[dd])");
    }
}
