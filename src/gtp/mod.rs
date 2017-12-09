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

use regex::Regex;
use rustyline::Editor;
use std::path::Path;

use go::{Board, Color};
use mcts;
use nn::Network;

mod vertex;

use gtp::vertex::*;

/// List containing all implemented commands, this is used to implement
/// the `list_commands` and `known_command` commands.
const KNOWN_COMMANDS: [&'static str; 14] = [
    "protocol_verion", "name", "version", "board_size", "clear_board", "komi", "play",
    "list_commands", "known_command", "showboard", "genmove", "reg_genmove", "undo",
    "quit"
];

#[derive(Debug, PartialEq)]
enum Command {
    Pass,  // do nothing
    ProtocolVersion,  // report protocol version
    Name,  // report the name of the program
    Version,  // report the version number of the program
    BoardSize(usize),  // set the board size to NxN
    ClearBoard,  // clear the board
    Komi(f32),  // set the komi
    Play(Color, Vertex),  // play a stone of the given color at the given vertex
    ListCommands,  // list all available commands
    KnownCommand(String),  // tell whether a command is known
    ShowBoard,  // write the position to stdout
    GenMove(Color),  // generate and play the supposedly best move for either color
    RegGenMove(Color),  // generate the supposedly best move for either color
    Undo,  // undo one move
    Quit  // quit
}

macro_rules! success {
    ($id:expr, $message:expr) => ({
        match $id {
            None => println!("= {}\n", $message),
            Some(id) => println!("={} {}\n", id, $message)
        }
    })
}

macro_rules! error {
    ($id:expr, $message:expr) => ({
        match $id {
            None => println!("? {}\n", $message),
            Some(id) => println!("?{} {}\n", id, $message)
        }
    })
}

lazy_static! {
    static ref ID_PREFIX: Regex = Regex::new(r"^([0-9]+)(?: +(.*)$|$)").unwrap();
    static ref BOARD_SIZE: Regex = Regex::new(r"^board_size +([0-9]+)").unwrap();
    static ref KOMI: Regex = Regex::new(r"^komi +([0-9\.]+)").unwrap();
    static ref PLAY: Regex = Regex::new(r"^play +([bBwW]) +([a-z][0-9]+)").unwrap();
    static ref KNOWN_COMMAND: Regex = Regex::new(r"^known_command +([^ ]+)").unwrap();
    static ref GENMOVE: Regex = Regex::new(r"^genmove +([bw])").unwrap();
    static ref REG_GENMOVE: Regex = Regex::new(r"^reg_genmove +([bBwW])").unwrap();
}

struct Gtp {
    network: Option<Network>,
    history: Vec<Board>,
    komi: f32
}

impl Gtp {
    /// Parse the GTP command in the given string and returns our internal
    /// representation of the given command.
    /// 
    /// # Arguments
    /// 
    /// * `id` -
    /// * `line` -
    /// 
    fn parse_command(id: Option<usize>, line: &str) -> Option<(Option<usize>, Command)> {
        let line = &line.to_lowercase();

        if line == "protocol_version" {
            Some((id, Command::ProtocolVersion))
        } else if line == "name" {
            Some((id, Command::Name))
        } else if line == "version" {
            Some((id, Command::Version))
        } else if let Some(caps) = BOARD_SIZE.captures(line) {
            let size = caps[1].parse::<usize>();

            if let Ok(size) = size {
                Some((id, Command::BoardSize(size)))
            } else {
                error!(id, "syntax error");
                Some((None, Command::Pass))
            }
        } else if line == "clear_board" {
            Some((id, Command::ClearBoard))
        } else if let Some(caps) = KOMI.captures(line) {
            let komi = caps[1].parse::<f32>();

            if let Ok(komi) = komi {
                Some((id, Command::Komi(komi)))
            } else {
                error!(id, "syntax error");
                Some((None, Command::Pass))
            }
        } else if let Some(caps) = PLAY.captures(line) {
            let color = caps[1].parse::<Color>();
            let vertex = caps[2].parse::<Vertex>();

            if let Ok(color) = color {
                if let Ok(vertex) = vertex {
                    Some((id, Command::Play(color, vertex)))
                } else {
                    error!(id, "syntax error");
                    Some((None, Command::Pass))
                }
            } else {
                error!(id, "syntax error");
                Some((None, Command::Pass))
            }
        } else if line == "list_commands" {
            Some((id, Command::ListCommands))
        } else if let Some(caps) = KNOWN_COMMAND.captures(line) {
            let command = &caps[1];

            Some((id, Command::KnownCommand(command.to_string())))
        } else if line == "showboard" {
            Some((id, Command::ShowBoard))
        } else if let Some(caps) = GENMOVE.captures(line) {
            let color = caps[1].parse::<Color>();

            if let Ok(color) = color {
                Some((id, Command::GenMove(color)))
            } else {
                error!(id, "syntax error");
                Some((None, Command::Pass))
            }
        } else if let Some(caps) = REG_GENMOVE.captures(line) {
            let color = caps[1].parse::<Color>();

            if let Ok(color) = color {
                Some((id, Command::RegGenMove(color)))
            } else {
                error!(id, "syntax error");
                Some((None, Command::Pass))
            }
        } else if line == "undo" {
            Some((id, Command::Undo))
        } else if line == "quit" {
            Some((id, Command::Quit))
        } else {
            error!(id, "unknown command");
            Some((None, Command::Pass))
        }
    }

    /// Parse the GTP command in the given string and returns our internal
    /// representation of the given command.
    /// 
    /// # Arguments
    /// 
    /// * `line` -
    /// 
    fn parse_line(line: &str) -> Option<(Option<usize>, Command)> {
        let line = line.trim();
        let line = {
            if let Some(pos) = line.find("#") {
                line[0..pos].to_string()
            } else {
                line.to_string()
            }
        };

        if let Some(caps) = ID_PREFIX.captures(&line) {
            let id = caps[1].parse::<usize>().unwrap();
            let rest = &caps[2];

            Gtp::parse_command(Some(id), rest.trim())
        } else {
            Gtp::parse_command(None, &line)
        }
    }

    fn generate_move(&mut self, id: Option<usize>, color: Color) -> Option<Vertex> {
        if self.network.is_none() {
            self.network = Network::new(Path::new("models/dream-go.json"));
        }

        let board = self.history.last().unwrap();

        if let Some(ref network) = self.network {
            let (value, index, _prior_index, _policy) = mcts::predict::<mcts::param::Standard, mcts::tree::DefaultValue>(network, &board, color);

            if value < -0.95 {  // 2.5% chance of winning
                success!(id, "resign");
                None
            } else if index >= 361 {  // passing move
                success!(id, "pass");
                None
            } else {
                let vertex = Vertex {
                    x: mcts::tree::X[index] as usize,
                    y: mcts::tree::Y[index] as usize
                };

                success!(id, &format!("{}", vertex));
                Some(vertex)
            }
        } else {
            error!(id, "unable to load network weights");

            None
        }
    }

    fn process(&mut self, id: Option<usize>, cmd: Command) {
        match cmd {
            Command::Quit => {}
            Command::Pass => {},
            Command::ProtocolVersion => { success!(id, "2"); },
            Command::Name => { success!(id, env!("CARGO_PKG_NAME")); },
            Command::Version => { success!(id, env!("CARGO_PKG_VERSION")); },
            Command::BoardSize(size) => {
                if size != 19 {
                    error!(id, "unacceptable size");
                } else {
                    self.history = vec! [Board::new()];
                    success!(id, "");
                }
            },
            Command::ClearBoard => {
                self.history = vec! [Board::new()];
                success!(id, "");
            },
            Command::Komi(komi) => {
                self.komi = komi;
                success!(id, "");
            },
            Command::Play(color, vertex) => {
                let next_board = {
                    let board = self.history.last().unwrap();

                    if vertex.is_pass() {
                        Some(board.clone())
                    } else if board.is_valid(color, vertex.x, vertex.y) {
                        let mut other = board.clone();
                        other.place(color, vertex.x, vertex.y);

                        Some(other)
                    } else {
                        None
                    }
                };

                if let Some(next_board) = next_board {
                    self.history.push(next_board);
                    success!(id, "");
                } else {
                    error!(id, "illegal move");
                }
            },
            Command::ListCommands => {
                success!(id, KNOWN_COMMANDS.join("\n"));
            },
            Command::KnownCommand(other) => {
                success!(id, {
                    if KNOWN_COMMANDS.iter().any(|&c| other == c) {
                        "true"
                    } else {
                        "false"
                    }
                });
            },
            Command::ShowBoard => {
                let board = self.history.last().unwrap();

                success!(id, &format!("{}", board));
            },
            Command::GenMove(color) => {
                let vertex = self.generate_move(id, color);

                if let Some(vertex) = vertex {
                    let mut board = self.history.last().unwrap().clone();
                    board.place(color, vertex.x, vertex.y);

                    self.history.push(board);
                }
            },
            Command::RegGenMove(color) => {
                self.generate_move(id, color);
            },
            Command::Undo => {
                if self.history.len() > 1 {
                    self.history.pop();
                    success!(id, "");
                } else {
                    error!(id, "cannot undo");
                }
            }
        }
    }
}

/// Run the GTP (Go Text Protocol) client that reads from standard input
/// and writes to standard output. This client implements the minimum
/// necessary feature-set of a GTP client.
pub fn run() {
    let mut rl = Editor::<()>::new();
    let mut gtp = Gtp {
        network: None,
        history: vec! [Board::new()],
        komi: 7.5
    };

    loop {
        if let Ok(line) = rl.readline("") {
            match Gtp::parse_line(&line) {
                Some((id, Command::Quit)) => {
                    success!(id, "");
                    break;
                },
                Some((id, cmd)) => gtp.process(id, cmd),
                _ => break
            }
        } else {
            break
        }
    }
}

#[cfg(test)]
mod tests {
    use go::*;
    use gtp::*;

    #[test]
    fn protocol_verion() {
        assert_eq!(Gtp::parse_line("1 protocol_version"), Some((Some(1), Command::ProtocolVersion)));
        assert_eq!(Gtp::parse_line("protocol_version"), Some((None, Command::ProtocolVersion)));
    }

    #[test]
    fn name() {
        assert_eq!(Gtp::parse_line("1 name"), Some((Some(1), Command::Name)));
        assert_eq!(Gtp::parse_line("name"), Some((None, Command::Name)));
    }

    #[test]
    fn version() {
        assert_eq!(Gtp::parse_line("1 version"), Some((Some(1), Command::Version)));
        assert_eq!(Gtp::parse_line("version"), Some((None, Command::Version)));
    }

    #[test]
    fn board_size() {
        assert_eq!(Gtp::parse_line("1 board_size 7"), Some((Some(1), Command::BoardSize(7))));
        assert_eq!(Gtp::parse_line("board_size 13"), Some((None, Command::BoardSize(13))));
    }

    #[test]
    fn clear_board() {
        assert_eq!(Gtp::parse_line("1 clear_board"), Some((Some(1), Command::ClearBoard)));
        assert_eq!(Gtp::parse_line("clear_board"), Some((None, Command::ClearBoard)));
    }

    #[test]
    fn komi() {
        assert_eq!(Gtp::parse_line("1 komi 0.5"), Some((Some(1), Command::Komi(0.5))));
        assert_eq!(Gtp::parse_line("komi 10"), Some((None, Command::Komi(10.0))));
    }

    #[test]
    fn play() {
        assert_eq!(Gtp::parse_line("1 play b c2"), Some((Some(1), Command::Play(Color::Black, Vertex{x: 2, y: 1}))));
        assert_eq!(Gtp::parse_line("play w a1"), Some((None, Command::Play(Color::White, Vertex{x: 0, y: 0}))));
    }

    #[test]
    fn list_commands() {
        assert_eq!(Gtp::parse_line("1 list_commands"), Some((Some(1), Command::ListCommands)));
        assert_eq!(Gtp::parse_line("list_commands"), Some((None, Command::ListCommands)));
    }

    #[test]
    fn known_command() {
        assert_eq!(Gtp::parse_line("1 known_command aaaa"), Some((Some(1), Command::KnownCommand("aaaa".to_string()))));
        assert_eq!(Gtp::parse_line("known_command genmove"), Some((None, Command::KnownCommand("genmove".to_string()))));
    }

    #[test]
    fn showboard() {
        assert_eq!(Gtp::parse_line("1 showboard"), Some((Some(1), Command::ShowBoard)));
        assert_eq!(Gtp::parse_line("showboard"), Some((None, Command::ShowBoard)));
    }

    #[test]
    fn genmove() {
        assert_eq!(Gtp::parse_line("1 genmove b"), Some((Some(1), Command::GenMove(Color::Black))));
        assert_eq!(Gtp::parse_line("genmove w"), Some((None, Command::GenMove(Color::White))));
    }

    #[test]
    fn reg_genmove() {
        assert_eq!(Gtp::parse_line("1 reg_genmove b"), Some((Some(1), Command::RegGenMove(Color::Black))));
        assert_eq!(Gtp::parse_line("reg_genmove w"), Some((None, Command::RegGenMove(Color::White))));
    }

    #[test]
    fn undo() {
        assert_eq!(Gtp::parse_line("1 undo"), Some((Some(1), Command::Undo)));
        assert_eq!(Gtp::parse_line("undo"), Some((None, Command::Undo)));
    }

    #[test]
    fn quit() {
        assert_eq!(Gtp::parse_line("1 quit"), Some((Some(1), Command::Quit)));
        assert_eq!(Gtp::parse_line("quit"), Some((None, Command::Quit)));
    }
}