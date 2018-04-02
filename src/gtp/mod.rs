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

use ordered_float::*;
use regex::Regex;
use std::io::BufRead;
use std::time::Instant;

use go::sgf::*;
use go::{Board, Color, Score};
use mcts::predict::{self, PredictService};
use mcts::time_control;
use mcts;
use nn::Network;
use util::config;

mod vertex;

use gtp::vertex::*;

/// List containing all implemented commands, this is used to implement
/// the `list_commands` and `known_command` commands.
const KNOWN_COMMANDS: [&'static str; 19] = [
    "protocol_verion", "name", "version", "boardsize", "clear_board", "komi", "play",
    "list_commands", "known_command", "showboard", "genmove", "reg_genmove", "undo",
    "time_settings", "quit", "final_score", "heatmap", "heatmap-nn",
    "sabaki-genmovelog"
];

#[derive(Debug, PartialEq)]
enum Command {
    Pass,  // do nothing
    ProtocolVersion,  // report protocol version
    Name,  // report the name of the program
    Version,  // report the version number of the program
    BoardSize(usize),  // set the board size to NxN
    ClearBoard,  // clear the board
    Heatmap(Color),  // sabaki heatmap for the given color
    HeatmapPrior(Color),  // sabaki heatmap (of the prior value) for the given color
    Komi(f32),  // set the komi
    Play(Color, Vertex),  // play a stone of the given color at the given vertex
    ListCommands,  // list all available commands
    KnownCommand(String),  // tell whether a command is known
    ShowBoard,  // write the position to stdout
    GenMove(Color),  // generate and play the supposedly best move for either color
    GenMoveLog,  // output all variations considered by the most recent search
    FinalScore,  // write the score to stdout
    RegGenMove(Color),  // generate the supposedly best move for either color
    Undo,  // undo one move
    TimeSettings(f32, f32, usize),  // set the time settings
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
    static ref BOARD_SIZE: Regex = Regex::new(r"^boardsize +([0-9]+)").unwrap();
    static ref HEATMAP: Regex = Regex::new(r"^heatmap +([bw])").unwrap();
    static ref HEATMAP_PRIOR: Regex = Regex::new(r"^heatmap-nn +([bw])").unwrap();
    static ref KOMI: Regex = Regex::new(r"^komi +([0-9\.]+)").unwrap();
    static ref PLAY: Regex = Regex::new(r"^play +([bBwW]) +([a-z][0-9]+|pass)").unwrap();
    static ref KNOWN_COMMAND: Regex = Regex::new(r"^known_command +([^ ]+)").unwrap();
    static ref GENMOVE: Regex = Regex::new(r"^genmove +([bw])").unwrap();
    static ref REG_GENMOVE: Regex = Regex::new(r"^reg_genmove +([bBwW])").unwrap();
    static ref TIME_SETTINGS: Regex = Regex::new(r"^time_settings +([0-9]+\.?[0-9]*) +([0-9]+\.?[0-9]*) +([0-9]+)").unwrap();
}

struct Gtp {
    service: Option<PredictService>,
    search_tree: Option<mcts::tree::Node<mcts::tree::DefaultValue>>,
    last_log: String,
    history: Vec<Board>,
    komi: f32,
    main_time: f32,
    byo_yomi_time: f32
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
        } else if let Some(caps) = HEATMAP.captures(line) {
            let color = caps[1].parse::<Color>();

            if let Ok(color) = color {
                Some((id, Command::Heatmap(color)))
            } else {
                error!(id, "syntax error");
                Some((None, Command::Pass))
            }
        } else if let Some(caps) = HEATMAP_PRIOR.captures(line) {
            let color = caps[1].parse::<Color>();

            if let Ok(color) = color {
                Some((id, Command::HeatmapPrior(color)))
            } else {
                error!(id, "syntax error");
                Some((None, Command::Pass))
            }
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
        } else if line == "sabaki-genmovelog" {
            Some((id, Command::GenMoveLog))
        } else if line == "final_score" {
            Some((id, Command::FinalScore))
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
        } else if let Some(caps) = TIME_SETTINGS.captures(line) {
            let main_time = caps[1].parse::<f32>();
            let byo_yomi_time = caps[2].parse::<f32>();
            let byo_yomi_stones = caps[3].parse::<usize>();

            if let Ok(main_time) = main_time {
                if let Ok(byo_yomi_time) = byo_yomi_time {
                    if let Ok(byo_yomi_stones) = byo_yomi_stones {
                        Some((id, Command::TimeSettings(main_time, byo_yomi_time, byo_yomi_stones)))
                    } else {
                        error!(id, "syntax error");
                        Some((None, Command::Pass))
                    }
                } else {
                    error!(id, "syntax error");
                    Some((None, Command::Pass))
                }
            } else {
                error!(id, "syntax error");
                Some((None, Command::Pass))
            }
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

        if line.is_empty() {
            Some((None, Command::Pass))
        } else if let Some(caps) = ID_PREFIX.captures(&line) {
            let id = caps[1].parse::<usize>().unwrap();
            let rest = &caps[2];

            Gtp::parse_command(Some(id), rest.trim())
        } else {
            Gtp::parse_command(None, &line)
        }
    }

    /// Create the `PredictService` if it does not exist, and then returns the
    /// current service.
    fn open_service(&mut self) -> &Option<PredictService> {
        if self.service.is_none() {
            match Network::new() {
                None => {},
                Some(network) => {
                    self.service = Some(predict::service(network));
                }
            }
        }

        &self.service
    }

    /// Generate a move using the monte carlo tree search engine for the given
    /// color, using the stored search tree if available.
    /// 
    /// If the given `color` is not the players whose turn it is according to the
    /// search tree then the tree is fast-forwarded until it is that players turn.
    /// 
    /// # Arguments
    /// 
    /// * `id` - the identifier of the command
    /// * `color` - the color to generate the move for
    /// 
    fn generate_move(&mut self, id: Option<usize>, color: Color) -> Option<Vertex> {
        self.open_service();

        if let Some(ref service) = self.service {
            let board = self.history.last().unwrap();
            let search_tree = self.search_tree.take().and_then(|tree| {
                if tree.color != color {
                    mcts::tree::Node::forward(tree, 361)  // pass
                } else {
                    Some(tree)
                }
            });

            let (value, index, tree) = if self.main_time.is_finite() && self.byo_yomi_time.is_finite() {
                let total_visits = search_tree.as_ref()
                    .map(|tree| tree.total_count)
                    .unwrap_or(0);

                mcts::predict::<mcts::tree::DefaultValue, _>(
                    &service.lock(),
                    None,
                    time_control::ByoYomi::new(board.count(), total_visits, self.main_time, self.byo_yomi_time),
                    search_tree,
                    &board,
                    color
                )
            } else {
                mcts::predict::<mcts::tree::DefaultValue, _>(
                    &service.lock(),
                    None,
                    time_control::RolloutLimit::new(*config::NUM_ROLLOUT),
                    search_tree,
                    &board,
                    color
                )
            };

            eprintln!("{}", mcts::tree::to_pretty(&tree));

            if !*config::NO_SABAKI {
                self.last_log = format!("{}", mcts::tree::to_sgf::<Sabaki, _>(&tree, &board, false));
            }
            self.search_tree = Some(tree);

            if value < 0.025 {  // 2.5% chance of winning
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

    /// Output all variations that were considered in the most recent search
    /// tree.
    /// 
    /// # Arguments
    /// 
    /// * `id` -
    /// 
    fn generate_move_log(&mut self, id: Option<usize>) {
        success!(id, &format!("#sabaki{{\"variations\":\"{}\"}}", self.last_log));
    }

    /// Returns a Sabaki heatmap that represents the given softmax policy.
    /// 
    /// # Arguments
    /// 
    /// * `softmax` -
    /// 
    fn to_heatmap(softmax: &[f32]) -> String {
        // format the flat softmax policy as a nested list (in JSON), where
        // each list correspond to one row on the board. The elements in the
        // inner list is the heat of each vertex, discretized to an integer
        // in `0..9`.
        let mut json = String::new();
        let max_heat = softmax.iter().take(361)
            .max_by_key(|&&v| OrderedFloat(v))
            .unwrap();

        for (index, _heat) in softmax.iter().take(361).enumerate() {
            let y = index / 19;
            let x = index % 19;

            if x == 0 {
                if y > 0 {
                    json += "],";
                }

                json += "[";
            }

            if x > 0 {
                json += ",";
            }

            // the GTP coordinates go from the bottom-left to the top-right, but
            // Sabaki heatmap coordinates go from the top-left to the
            // bottom-right (...) so inverse the y-axis.
            let other = softmax[19 * (18 - y) + x];

            json += &format!("{}", (9.0 * other / max_heat).ceil());
        }

        format!("{{\"heatmap\":[{}]]}}", json)
    }

    /// Generate a move using the engine and then output an Sabaki GTP extension
    /// string that allows us to draw the heatmap directly on the board.
    /// 
    /// # Arguments
    /// 
    /// * `id` -
    /// * `color` -
    /// * `prior` - whether to show the _prior_ as the heatmap
    /// 
    fn heatmap(&mut self, id: Option<usize>, color: Color, prior: bool) {
        self.open_service();

        if let Some(ref service) = self.service {
            let board = self.history.last().unwrap();
            let (_value, _index, tree) = mcts::predict::<mcts::tree::DefaultValue, _>(
                &service.lock(),
                None,
                time_control::RolloutLimit::new(*config::NUM_ROLLOUT),
                None,
                &board,
                color
            );

            eprintln!("{}", mcts::tree::to_pretty(&tree));

            // output the heatmap in Sabaki format
            let json = if prior {
                let mut s = vec! [0.0f32; 362];
                let mut s_total = 0.0f32;

                for i in 0..362 {
                    if tree.prior[i].is_finite() {
                        s_total += tree.prior[i];
                    }
                }

                for i in 0..362 {
                    if tree.prior[i].is_finite() {
                        s[i] = tree.prior[i] / s_total;
                    }
                }

                Gtp::to_heatmap(&s)
            } else {
                Gtp::to_heatmap(&tree.softmax())
            };

            success!(id, &format!("#sabaki{}", json));
        } else {
            error!(id, "unable to load network weights");
        }
    }

    fn process(&mut self, id: Option<usize>, cmd: Command) {
        match cmd {
            Command::Quit => {}
            Command::Pass => {},
            Command::ProtocolVersion => { success!(id, "2"); },
            Command::Name => {
                success!(id, config::get_env::<String>("DG_NAME")
                    .unwrap_or(env!("CARGO_PKG_NAME").to_string())
                );
            },
            Command::Version => {
                success!(id, config::get_env::<String>("DG_VERSION")
                    .unwrap_or(env!("CARGO_PKG_VERSION").to_string())
                );
            },
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
                self.search_tree = None;
                success!(id, "");
            },
            Command::Komi(komi) => {
                self.komi = komi;
                success!(id, "");
            },
            Command::Heatmap(color) => {
                self.heatmap(id, color, false);
            },
            Command::HeatmapPrior(color) => {
                self.heatmap(id, color, true);
            },
            Command::Play(color, vertex) => {
                let next_board = {
                    let board = self.history.last().unwrap();

                    if vertex.is_pass() {
                        self.search_tree = self.search_tree.take().and_then(|tree| {
                            if tree.color == color {
                                mcts::tree::Node::forward(tree, 361)
                            } else if let Some(tree) = mcts::tree::Node::forward(tree, 361) {
                                mcts::tree::Node::forward(tree, 361)
                            } else {
                                None
                            }
                        });

                        Some(board.clone())
                    } else if board.is_valid(color, vertex.x, vertex.y) {
                        let mut other = board.clone();
                        other.place(color, vertex.x, vertex.y);
                        self.search_tree = self.search_tree.take().and_then(|tree| {
                            let index = 19 * vertex.y + vertex.x;

                            if tree.color == color {
                                mcts::tree::Node::forward(tree, index)
                            } else if let Some(tree) = mcts::tree::Node::forward(tree, 361) {
                                // if it is not that players turn, then there is an implied
                                // passing move from the opponent
                                mcts::tree::Node::forward(tree, index)
                            } else {
                                None
                            }
                        });

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
                let known_commands = KNOWN_COMMANDS.iter()
                    .cloned()
                    .filter(|&c| {
                        !c.starts_with("sabaki-") || !*config::NO_SABAKI
                    }).collect::<Vec<&str>>();

                success!(id, known_commands.join("\n"));
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

                success!(id, &format!("\n{}", board));
            },
            Command::GenMove(color) => {
                let start_time = Instant::now();
                let vertex = self.generate_move(id, color);

                if let Some(vertex) = vertex {
                    let mut board = self.history.last().unwrap().clone();
                    board.place(color, vertex.x, vertex.y);

                    self.history.push(board);
                    self.search_tree = self.search_tree.take().and_then(|tree| {
                        mcts::tree::Node::forward(tree, 19 * vertex.y + vertex.x)
                    });
                } else {
                    self.search_tree = self.search_tree.take().and_then(|tree| {
                        mcts::tree::Node::forward(tree, 361)
                    });
                }

                // update the remaining main time, saturating at zero instead of
                // overflowing.
                let duration = start_time.elapsed();

                self.main_time -= duration.as_secs() as f32;
                self.main_time -= duration.subsec_nanos() as f32 / 1e9;

                if self.main_time < 0.0 {
                    self.main_time = 0.0;
                }
            },
            Command::GenMoveLog => {
                self.generate_move_log(id);
            },
            Command::FinalScore => {
                let board = self.history.last().unwrap();
                let (black, white, rollout) = board.get_guess_score();
                let black = black as f32;
                let white = white as f32 + self.komi;

                if !*config::NO_SABAKI {
                    self.last_log = format!("({})", rollout);
                }

                if black == white {
                    success!(id, "0");
                } else if black > white {
                    success!(id, &format!("B+{:.1}", black - white));
                } else if white > black {
                    success!(id, &format!("W+{:.1}", white - black));
                }
            }
            Command::RegGenMove(color) => {
                self.search_tree = None;
                self.generate_move(id, color);
            },
            Command::Undo => {
                if self.history.len() > 1 {
                    self.history.pop();
                    self.search_tree = None;
                    success!(id, "");
                } else {
                    error!(id, "cannot undo");
                }
            },
            Command::TimeSettings(main_time, byo_yomi_time, byo_yomi_stones) => {
                // ensure the neural network weights are loaded since we do not
                // want that to be part of the allocated time
                self.open_service();

                if byo_yomi_stones == 0 {
                    self.main_time = ::std::f32::INFINITY;
                    self.byo_yomi_time = ::std::f32::INFINITY;
                    success!(id, "");
                } else if byo_yomi_stones == 1 {
                    if main_time >= 0.0 && byo_yomi_time >= 0.0 {
                        self.main_time = main_time;
                        self.byo_yomi_time = byo_yomi_time;
                        success!(id, "");
                    } else {
                        error!(id, "syntax error");
                    }
                } else {
                    error!(id, "syntax error");
                }
            }
        }
    }
}

/// Run the GTP (Go Text Protocol) client that reads from standard input
/// and writes to standard output. This client implements the minimum
/// necessary feature-set of a GTP client.
pub fn run() {
    let stdin = ::std::io::stdin();
    let stdin_lock = stdin.lock();
    let mut gtp = Gtp {
        service: None,
        search_tree: None,
        last_log: "{}".to_string(),
        history: vec! [Board::new()],
        komi: 7.5,
        main_time: ::std::f32::INFINITY,
        byo_yomi_time: ::std::f32::INFINITY
    };

    for line in stdin_lock.lines() {
        if let Ok(line) = line {
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
    fn boardsize() {
        assert_eq!(Gtp::parse_line("1 boardsize 7"), Some((Some(1), Command::BoardSize(7))));
        assert_eq!(Gtp::parse_line("boardsize 13"), Some((None, Command::BoardSize(13))));
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
    fn final_score() {
        assert_eq!(Gtp::parse_line("1 final_score"), Some((Some(1), Command::FinalScore)));
        assert_eq!(Gtp::parse_line("final_score"), Some((None, Command::FinalScore)));
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
    fn time_settings() {
        assert_eq!(Gtp::parse_line("1 time_settings 30.2 0 0"), Some((Some(1), Command::TimeSettings(30.2, 0.0, 0))));
        assert_eq!(Gtp::parse_line("time_settings 300 3.14 0"), Some((None, Command::TimeSettings(300.0, 3.14, 0))));
    }

    #[test]
    fn quit() {
        assert_eq!(Gtp::parse_line("1 quit"), Some((Some(1), Command::Quit)));
        assert_eq!(Gtp::parse_line("quit"), Some((None, Command::Quit)));
    }

    #[test]
    fn empty() {
        assert_eq!(Gtp::parse_line(""), Some((None, Command::Pass)));
    }
}
