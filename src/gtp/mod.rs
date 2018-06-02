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
use go::{Board, Color, Score, StoneStatus};
use mcts::predict::{self, PredictService};
use mcts::time_control;
use mcts;
use nn::Network;
use util::config;

mod time_settings;
mod vertex;

use gtp::vertex::*;

/// List containing all implemented commands, this is used to implement
/// the `list_commands` and `known_command` commands.
const KNOWN_COMMANDS: [&'static str; 23] = [
    "protocol_verion", "name", "version", "boardsize", "clear_board", "komi", "play",
    "list_commands", "known_command", "showboard", "genmove", "reg_genmove",
    "kgs-genmove_cleanup", "undo",
    "time_settings", "kgs-time_settings", "time_left", "quit",
    "final_score", "final_status_list",
    "heatmap", "heatmap-nn", "sabaki-genmovelog"
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
    GenMove(Color, bool),  // generate and play the supposedly best move for either color, the second argument indicate whether it is a clean-up move
    GenMoveLog,  // output all variations considered by the most recent search
    FinalScore,  // write the score to stdout
    FinalStatusList(StoneStatus),  // write status of stones to stdout
    RegGenMove(Color),  // generate the supposedly best move for either color
    Undo,  // undo one move
    TimeSettingsNone,  // set the time settings
    TimeSettingsAbsolute(f32),  // set the time settings
    TimeSettingsCanadian(f32, f32, usize),  // set the time settings
    TimeSettingsByoYomi(f32, f32, usize),  // set the time settings
    TimeLeft(Color, f32, usize),  // set the remaining time for the given color
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
    static ref KGS_GENMOVE_CLEANUP: Regex = Regex::new(r"^kgs-genmove_cleanup +([bw])").unwrap();
    static ref FINAL_STATUS_LIST: Regex = Regex::new(r"^final_status_list +(dead|alive|seki)").unwrap();
    static ref TIME_SETTINGS: Regex = Regex::new(r"^time_settings +([0-9]+\.?[0-9]*) +([0-9]+\.?[0-9]*) +([0-9]+)").unwrap();
    static ref KGS_TIME_SETTINGS_NONE: Regex = Regex::new(r"^kgs-time_settings +none").unwrap();
    static ref KGS_TIME_SETTINGS_ABSOLUTE: Regex = Regex::new(r"^kgs-time_settings +absolute +([0-9]+\.?[0-9]*)").unwrap();
    static ref KGS_TIME_SETTINGS_BYOYOMI: Regex = Regex::new(r"^kgs-time_settings +byoyomi +([0-9]+\.?[0-9]*) +([0-9]+\.?[0-9]*) +([0-9]+)").unwrap();
    static ref KGS_TIME_SETTINGS_CANADIAN: Regex = Regex::new(r"^kgs-time_settings +canadian +([0-9]+\.?[0-9]*) +([0-9]+\.?[0-9]*) +([0-9]+)").unwrap();
    static ref TIME_LEFT: Regex = Regex::new(r"^time_left +([bBwW]) +([0-9]+\.?[0-9]*) +([0-9]+)").unwrap();
}

struct Gtp {
    service: Option<PredictService>,
    search_tree: Option<mcts::tree::Node<mcts::tree::DefaultValue>>,
    last_log: String,
    history: Vec<Board>,
    komi: f32,
    time_settings: [Box<time_settings::TimeSettings>; 3]
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
    fn parse_command(id: Option<usize>, line: &str) -> Result<(Option<usize>, Command), &str> {
        let line = &line.to_lowercase();

        if line == "protocol_version" {
            Ok((id, Command::ProtocolVersion))
        } else if line == "name" {
            Ok((id, Command::Name))
        } else if line == "version" {
            Ok((id, Command::Version))
        } else if let Some(caps) = BOARD_SIZE.captures(line) {
            let size = caps[1].parse::<usize>().map_err(|_| "syntax error")?;

            Ok((id, Command::BoardSize(size)))
        } else if line == "clear_board" {
            Ok((id, Command::ClearBoard))
        } else if let Some(caps) = HEATMAP.captures(line) {
            let color = caps[1].parse::<Color>().map_err(|_| "syntax error")?;

            Ok((id, Command::Heatmap(color)))
        } else if let Some(caps) = HEATMAP_PRIOR.captures(line) {
            let color = caps[1].parse::<Color>().map_err(|_| "syntax error")?;

            Ok((id, Command::HeatmapPrior(color)))
        } else if let Some(caps) = KOMI.captures(line) {
            let komi = caps[1].parse::<f32>().map_err(|_| "syntax error")?;

            Ok((id, Command::Komi(komi)))
        } else if let Some(caps) = PLAY.captures(line) {
            let color = caps[1].parse::<Color>().map_err(|_| "syntax error")?;
            let vertex = caps[2].parse::<Vertex>().map_err(|_| "syntax error")?;

            Ok((id, Command::Play(color, vertex)))
        } else if line == "list_commands" {
            Ok((id, Command::ListCommands))
        } else if let Some(caps) = KNOWN_COMMAND.captures(line) {
            let command = &caps[1];

            Ok((id, Command::KnownCommand(command.to_string())))
        } else if line == "showboard" {
            Ok((id, Command::ShowBoard))
        } else if let Some(caps) = GENMOVE.captures(line) {
            let color = caps[1].parse::<Color>().map_err(|_| "syntax error")?;

            Ok((id, Command::GenMove(color, false)))
        } else if line == "sabaki-genmovelog" {
            Ok((id, Command::GenMoveLog))
        } else if line == "final_score" {
            Ok((id, Command::FinalScore))
        } else if let Some(caps) = FINAL_STATUS_LIST.captures(line) {
            let status = caps[1].parse::<StoneStatus>().map_err(|_| "syntax error")?;

            Ok((id, Command::FinalStatusList(status)))
        } else if let Some(caps) = REG_GENMOVE.captures(line) {
            let color = caps[1].parse::<Color>().map_err(|_| "syntax error")?;

            Ok((id, Command::RegGenMove(color)))
        } else if let Some(caps) = KGS_GENMOVE_CLEANUP.captures(line) {
            let color = caps[1].parse::<Color>().map_err(|_| "syntax error")?;

            Ok((id, Command::GenMove(color, true)))
        } else if line == "undo" {
            Ok((id, Command::Undo))
        } else if let Some(caps) = TIME_SETTINGS.captures(line) {
            let main_time = caps[1].parse::<f32>().map_err(|_| "syntax error")?;
            let byo_yomi_time = caps[2].parse::<f32>().map_err(|_| "syntax error")?;
            let byo_yomi_stones = caps[3].parse::<usize>().map_err(|_| "syntax error")?;

            if byo_yomi_stones == 0 {
                Ok((id, Command::TimeSettingsNone))
            } else {
                Ok((id, Command::TimeSettingsCanadian(main_time, byo_yomi_time, byo_yomi_stones)))
            }
        } else if let Some(_caps) = KGS_TIME_SETTINGS_NONE.captures(line) {
            Ok((id, Command::TimeSettingsNone))
        } else if let Some(caps) = KGS_TIME_SETTINGS_ABSOLUTE.captures(line) {
            let main_time = caps[1].parse::<f32>().map_err(|_| "syntax error")?;

            Ok((id, Command::TimeSettingsAbsolute(main_time)))
        } else if let Some(caps) = KGS_TIME_SETTINGS_BYOYOMI.captures(line) {
            let main_time = caps[1].parse::<f32>().map_err(|_| "syntax error")?;
            let byo_yomi_time = caps[2].parse::<f32>().map_err(|_| "syntax error")?;
            let byo_yomi_stones = caps[3].parse::<usize>().map_err(|_| "syntax error")?;

            Ok((id, Command::TimeSettingsByoYomi(main_time, byo_yomi_time, byo_yomi_stones)))
        } else if let Some(caps) = KGS_TIME_SETTINGS_CANADIAN.captures(line) {
            let main_time = caps[1].parse::<f32>().map_err(|_| "syntax error")?;
            let byo_yomi_time = caps[2].parse::<f32>().map_err(|_| "syntax error")?;
            let byo_yomi_stones = caps[3].parse::<usize>().map_err(|_| "syntax error")?;

            if byo_yomi_stones > 0 {
                Ok((id, Command::TimeSettingsCanadian(main_time, byo_yomi_time, byo_yomi_stones)))
            } else {
                Err("syntax error")
            }
        } else if let Some(caps) = TIME_LEFT.captures(line) {
            let color = caps[1].parse::<Color>().map_err(|_| "syntax error")?;
            let main_time = caps[2].parse::<f32>().map_err(|_| "syntax error")?;
            let byo_yomi_stones = caps[3].parse::<usize>().map_err(|_| "syntax error")?;

            Ok((id, Command::TimeLeft(color, main_time, byo_yomi_stones)))
        } else if line == "quit" {
            Ok((id, Command::Quit))
        } else {
            error!(id, "unknown command");
            Ok((None, Command::Pass))
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

            match Gtp::parse_command(Some(id), rest.trim()) {
                Ok(result) => Some(result),
                Err(reason) => {
                    error!(Some(id), reason);
                    Some((None, Command::Pass))
                }
            }
        } else {
            match Gtp::parse_command(None, &line) {
                Ok(result) => Some(result),
                Err(reason) => {
                    error!(None as Option<usize>, reason);

                    Some((None, Command::Pass))
                }
            }
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
    /// * `is_cleanup` - determine whether this is a clean-up move
    /// 
    fn generate_move(&mut self, id: Option<usize>, color: Color, is_cleanup: bool) -> Option<Vertex> {
        self.open_service();

        if let Some(ref service) = self.service {
            let board = self.history.last().unwrap();
            let mut search_tree = self.search_tree.take().and_then(|tree| {
                if tree.color != color {
                    mcts::tree::Node::forward(tree, 361)  // pass
                } else {
                    Some(tree)
                }
            });

            // disqualify the `pass` move if we are doing clean-up and the board
            // is not scoreable.
            if is_cleanup && !board.is_scoreable() {
                if let Some(ref mut search_tree) = search_tree {
                    search_tree.disqualify(361);
                }
            }

            let (main_time, byo_yomi_time, byo_yomi_periods) = self.time_settings[color as usize].remaining();
            let (value, index, tree) = if main_time.is_finite() && byo_yomi_time.is_finite() {
                let total_visits = search_tree.as_ref()
                    .map(|tree| tree.total_count)
                    .unwrap_or(0);

                mcts::predict::<mcts::tree::DefaultValue, _>(
                    &service.lock(),
                    None,
                    time_control::ByoYomi::new(board.count(), total_visits, main_time, byo_yomi_time, byo_yomi_periods),
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

            if value < 0.1 {  // 10% chance of winning
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
            Command::GenMove(color, is_cleanup) => {
                let start_time = Instant::now();
                let vertex = self.generate_move(id, color, is_cleanup);

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
                let elapsed = start_time.elapsed();
                let elapsed_secs = elapsed.as_secs() as f32 + elapsed.subsec_nanos() as f32 / 1e9;
                let c = color as usize;

                self.time_settings[c].update(elapsed_secs);
            },
            Command::GenMoveLog => {
                self.generate_move_log(id);
            },
            Command::FinalScore => {
                self.open_service();

                if let Some(ref service) = self.service {
                    let board = self.history.last().unwrap();
                    let next_color = match board.last_played() {
                        Some(color) => color.opposite(),
                        _ => Color::Black,
                    };
                    let (finished, rollout) = mcts::greedy_score(&service.lock(), &board, next_color);
                    let (black, white) = board.get_guess_score(&finished);

                    eprintln!("Black: {}", black);
                    eprintln!("White: {} + {}", white, self.komi);

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
                } else {
                    error!(id, "");
                }
            },
            Command::FinalStatusList(status) => {
                self.open_service();

                if let Some(ref service) = self.service {
                    let board = self.history.last().unwrap();
                    let next_color = match board.last_played() {
                        Some(color) => color.opposite(),
                        _ => Color::Black,
                    };
                    let (finished, _rollout) = mcts::greedy_score(&service.lock(), &board, next_color);
                    let status_list = board.get_stone_status(&finished);
                    let vertices = status_list.into_iter()
                        .filter_map(|(index, stone_status)| {
                            if stone_status == status {
                                let vertex = Vertex {
                                    x: mcts::tree::X[index] as usize,
                                    y: mcts::tree::Y[index] as usize
                                };

                                Some(format!("{}", vertex))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<String>>();

                    success!(id, vertices.join(" "));
                } else {
                    error!(id, "could not load network weights");
                }
            },
            Command::RegGenMove(color) => {
                self.search_tree = None;
                self.generate_move(id, color, false);
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
            Command::TimeSettingsNone => {
                // ensure the neural network weights are loaded since we do not
                // want that to be part of the allocated time
                self.open_service();

                for &c in &[Color::Black, Color::White] {
                    self.time_settings[c as usize] = Box::new(time_settings::None::new());
                }

                success!(id, "");
            },
            Command::TimeSettingsAbsolute(main_time) => {
                // ensure the neural network weights are loaded since we do not
                // want that to be part of the allocated time
                self.open_service();

                for &c in &[Color::Black, Color::White] {
                    self.time_settings[c as usize] = Box::new(time_settings::Absolute::new(main_time));
                }

                success!(id, "");
            },
            Command::TimeSettingsByoYomi(main_time, byo_yomi_time, byo_yomi_stones) => {
                // ensure the neural network weights are loaded since we do not
                // want that to be part of the allocated time
                self.open_service();

                for &c in &[Color::Black, Color::White] {
                    self.time_settings[c as usize] = Box::new(time_settings::ByoYomi::new(
                        main_time,
                        byo_yomi_time,
                        byo_yomi_stones
                    ));
                }

                success!(id, "");
            },
            Command::TimeSettingsCanadian(main_time, byo_yomi_time, byo_yomi_stones) => {
                // ensure the neural network weights are loaded since we do not
                // want that to be part of the allocated time
                self.open_service();

                for &c in &[Color::Black, Color::White] {
                    self.time_settings[c as usize] = Box::new(time_settings::Canadian::new(
                        main_time,
                        byo_yomi_time,
                        byo_yomi_stones
                    ));
                }

                success!(id, "");
            },
            Command::TimeLeft(color, main_time, byo_yomi_stones) => {
                let c = color as usize;

                // ensure the neural network weights are loaded since we do not
                // want that to be part of the allocated time
                self.open_service();
                self.time_settings[c].time_left(main_time, byo_yomi_stones);
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
        time_settings: [
            Box::new(time_settings::None::new()),
            Box::new(time_settings::None::new()),
            Box::new(time_settings::None::new()),
        ],
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
        assert_eq!(Gtp::parse_line("1 genmove b"), Some((Some(1), Command::GenMove(Color::Black, false))));
        assert_eq!(Gtp::parse_line("genmove w"), Some((None, Command::GenMove(Color::White, false))));
    }

    #[test]
    fn final_score() {
        assert_eq!(Gtp::parse_line("1 final_score"), Some((Some(1), Command::FinalScore)));
        assert_eq!(Gtp::parse_line("final_score"), Some((None, Command::FinalScore)));
    }

    #[test]
    fn final_status_list() {
        assert_eq!(Gtp::parse_line("1 final_status_list dead"), Some((Some(1), Command::FinalStatusList(StoneStatus::Dead))));
        assert_eq!(Gtp::parse_line("final_status_list alive"), Some((None, Command::FinalStatusList(StoneStatus::Alive))));
        assert_eq!(Gtp::parse_line("final_status_list dead"), Some((None, Command::FinalStatusList(StoneStatus::Dead))));
        assert_eq!(Gtp::parse_line("final_status_list seki"), Some((None, Command::FinalStatusList(StoneStatus::Seki))));
    }

    #[test]
    fn reg_genmove() {
        assert_eq!(Gtp::parse_line("1 reg_genmove b"), Some((Some(1), Command::RegGenMove(Color::Black))));
        assert_eq!(Gtp::parse_line("reg_genmove w"), Some((None, Command::RegGenMove(Color::White))));
    }

    #[test]
    fn kgs_genmove_cleanup() {
        assert_eq!(Gtp::parse_line("1 kgs-genmove_cleanup b"), Some((Some(1), Command::GenMove(Color::Black, true))));
        assert_eq!(Gtp::parse_line("kgs-genmove_cleanup w"), Some((None, Command::GenMove(Color::White, true))));
    }

    #[test]
    fn undo() {
        assert_eq!(Gtp::parse_line("1 undo"), Some((Some(1), Command::Undo)));
        assert_eq!(Gtp::parse_line("undo"), Some((None, Command::Undo)));
    }

    #[test]
    fn time_settings() {
        assert_eq!(Gtp::parse_line("1 time_settings 30.2 0 0"), Some((Some(1), Command::TimeSettingsNone)));
        assert_eq!(Gtp::parse_line("time_settings 300 3.14 1"), Some((None, Command::TimeSettingsCanadian(300.0, 3.14, 1))));
    }

    #[test]
    fn kgs_time_settings() {
        assert_eq!(Gtp::parse_line("1 kgs-time_settings none"), Some((Some(1), Command::TimeSettingsNone)));
        assert_eq!(Gtp::parse_line("kgs-time_settings none"), Some((None, Command::TimeSettingsNone)));

        assert_eq!(Gtp::parse_line("2 kgs-time_settings absolute 30.2"), Some((Some(2), Command::TimeSettingsAbsolute(30.2))));
        assert_eq!(Gtp::parse_line("kgs-time_settings absolute 300"), Some((None, Command::TimeSettingsAbsolute(300.0))));

        assert_eq!(Gtp::parse_line("3 kgs-time_settings byoyomi 30.2 0 0"), Some((Some(3), Command::TimeSettingsByoYomi(30.2, 0.0, 0))));
        assert_eq!(Gtp::parse_line("kgs-time_settings byoyomi 300 3.14 1"), Some((None, Command::TimeSettingsByoYomi(300.0, 3.14, 1))));

        assert_eq!(Gtp::parse_line("4 kgs-time_settings canadian 30.2 1 1"), Some((Some(4), Command::TimeSettingsCanadian(30.2, 1.0, 1))));
        assert_eq!(Gtp::parse_line("kgs-time_settings canadian 300 3.14 1"), Some((None, Command::TimeSettingsCanadian(300.0, 3.14, 1))));
    }

    #[test]
    fn time_left() {
        assert_eq!(Gtp::parse_line("1 time_left b 3.14 0"), Some((Some(1), Command::TimeLeft(Color::Black, 3.14, 0))));
        assert_eq!(Gtp::parse_line("time_left W 278.1 1"), Some((None, Command::TimeLeft(Color::White, 278.1, 1))));
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
