// Copyright 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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
use std::io::BufRead;
use std::time::Instant;

use go::util::score::{Score, StoneStatus};
use go::{DEFAULT_KOMI, Board, Color};
use mcts::time_control;
use mcts;
use util::config;

mod ponder_service;
mod time_settings;
mod vertex;

use gtp::vertex::*;
use gtp::ponder_service::PonderService;

/// List containing all implemented commands, this is used to implement
/// the `list_commands` and `known_command` commands.
const KNOWN_COMMANDS: [&str; 20] = [
    "protocol_version", "name", "version", "boardsize", "clear_board", "komi", "play",
    "list_commands", "known_command", "showboard", "genmove", "reg_genmove",
    "kgs-genmove_cleanup", "undo",
    "time_settings", "kgs-time_settings", "time_left", "quit",
    "final_score", "final_status_list"
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
    GenMove(Color, bool),  // generate and play the supposedly best move for either color, the second argument indicate whether it is a clean-up move
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
    static ref KOMI: Regex = Regex::new(r"^komi +(-?[0-9\.]+)").unwrap();
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
    ponder: PonderService,
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

            Ok((id, Command::GenMove(color, *config::TROMP_TAYLOR)))
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

            if byo_yomi_time > 0.0 && byo_yomi_stones == 0 {
                // we gain extra time every zero stones, so infinite...
                Ok((id, Command::TimeSettingsNone))
            } else if byo_yomi_time == 0.0 {
                // this is effectively absolute time since we gain no extra
                // time.
                Ok((id, Command::TimeSettingsAbsolute(main_time)))
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
            if let Some(pos) = line.find('#') {
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
        let (main_time, byo_yomi_time, byo_yomi_periods) = self.time_settings[color as usize].remaining();
        let board = self.history.last().unwrap();
        let result = self.ponder.service(|service, search_tree, p_state| {
            let search_tree = if search_tree.color != color {
                // passing moves are not recorded in the GTP protocol, so we
                // will just assume the other player passed once if we are in
                // this situation
                mcts::tree::Node::forward(search_tree, 361)
            } else {
                Some(search_tree)
            };

            let result = if main_time.is_finite() && byo_yomi_time.is_finite() {
                let total_visits = search_tree.as_ref()
                    .map(|tree| tree.total_count)
                    .unwrap_or(0);

                mcts::predict(
                    &service.lock().clone_to_static(),
                    None,
                    time_control::ByoYomi::new(board.count(), total_visits, main_time, byo_yomi_time, byo_yomi_periods),
                    search_tree,
                    &board,
                    color
                )
            } else {
                mcts::predict(
                    &service.lock().clone_to_static(),
                    None,
                    time_control::RolloutLimit::new((*config::NUM_ROLLOUT).into()),
                    search_tree,
                    &board,
                    color
                )
            };

            if result.is_none() {
                return (None, None, p_state)
            }

            // disqualify the `pass` move, and any move that is not in contested territory, if
            // we are doing clean-up and the board is not scorable.
            let (value, index, mut tree) = result.unwrap();
            let (value, index) = if is_cleanup && index == 361 && !board.is_scorable() {
                tree.disqualify(361);

                for &index in &board.get_scorable_territory() {
                    tree.disqualify(index);
                }

                tree.best(0.0)
            } else {
                (value, index)
            };

            eprintln!("{}", mcts::tree::to_pretty(&tree));

            let should_resign = !*config::NO_RESIGN && value.is_finite() && value < 0.1;  // 10% chance of winning
            let index = if should_resign { 361 } else { index };
            let (vertex, tree, other) = if index >= 361 {  // passing move
                (None, mcts::tree::Node::forward(tree, 361), board.clone())
            } else {
                let (x, y) = (mcts::tree::X[index] as usize, mcts::tree::Y[index] as usize);
                let mut other = board.clone();

                other.place(color, x, y);
                (Some(Vertex { x, y }), mcts::tree::Node::forward(tree, index), other)
            };

            (Some((vertex, should_resign)), tree, (other, color.opposite()))
        });

        if let Ok(Some((vertex, should_resign))) = result {
            if should_resign {
                success!(id, "resign");
                None
            } else if let Some(vertex) = vertex {  // passing move
                success!(id, &format!("{}", vertex));
                Some(vertex)
            } else {
                success!(id, "pass");
                None
            }
        } else if let Ok(None) = result {
            error!(id, "unrecognized error");

            None
        } else {
            error!(id, result.err().unwrap());

            None
        }
    }

    fn process(&mut self, id: Option<usize>, cmd: Command) {
        match cmd {
            Command::Quit => {}
            Command::Pass => {},
            Command::ProtocolVersion => { success!(id, "2"); },
            Command::Name => {
                success!(id, config::get_env::<String>("DG_NAME")
                    .unwrap_or_else(|| env!("CARGO_PKG_NAME").to_string())
                );
            },
            Command::Version => {
                success!(id, config::get_env::<String>("DG_VERSION")
                    .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
                );
            },
            Command::BoardSize(size) => {
                if size != 19 {
                    error!(id, "unacceptable size");
                } else {
                    self.history = vec! [Board::new(self.komi)];
                    success!(id, "");
                }
            },
            Command::ClearBoard => {
                self.history = vec! [Board::new(self.komi)];
                self.ponder = PonderService::new(Board::new(self.komi), Color::Black);
                success!(id, "");
            },
            Command::Komi(komi) => {
                self.komi = komi;
                for board in self.history.iter_mut() {
                    (*board).set_komi(komi);
                }

                // restart the pondering service, since we have been thinking
                // with the wrong komi.
                let board = self.history.last().unwrap().clone();
                let next_color = match board.last_played() {
                    Some(color) => color.opposite(),
                    None => Color::Black
                };

                self.ponder = PonderService::new(board, next_color);

                success!(id, "");
            },
            Command::Play(color, vertex) => {
                let next_board = {
                    let board = self.history.last().unwrap();

                    if vertex.is_pass() {
                        self.ponder.forward(color, None);

                        Some(board.clone())
                    } else if board.is_valid(color, vertex.x, vertex.y) {
                        let mut other = board.clone();

                        other.place(color, vertex.x, vertex.y);
                        self.ponder.forward(color, Some((vertex.x, vertex.y)));

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

                success!(id, &format!("\n{}", board));
            },
            Command::GenMove(color, is_cleanup) => {
                let start_time = Instant::now();
                let vertex = self.generate_move(id, color, is_cleanup);

                if let Some(vertex) = vertex {
                    let mut board = self.history.last().unwrap().clone();
                    board.place(color, vertex.x, vertex.y);

                    self.history.push(board);
                }

                // update the remaining main time, saturating at zero instead of
                // overflowing.
                let elapsed = start_time.elapsed();
                let elapsed_secs = elapsed.as_secs() as f32 + elapsed.subsec_nanos() as f32 / 1e9;
                let c = color as usize;

                self.time_settings[c].update(elapsed_secs);
            },
            Command::FinalScore => {
                let board = self.history.last().unwrap();
                let result = self.ponder.service(|service, search_tree, p_state| {
                    let next_color = match board.last_played() {
                        Some(color) => color.opposite(),
                        _ => Color::Black,
                    };
                    let (finished, rollout) = mcts::greedy_score(&service.lock(), &board, next_color);
                    let (black, white) = board.get_guess_score(&finished);

                    ((black, white, rollout), Some(search_tree), p_state)
                });

                if let Ok((black, white, _rollout)) = result {
                    eprintln!("Black: {}", black);
                    eprintln!("White: {} + {}", white, self.komi);

                    let black = black as f32;
                    let white = white as f32 + self.komi;

                    if black == white {
                        success!(id, "0");
                    } else if black > white {
                        success!(id, &format!("B+{:.1}", black - white));
                    } else if white > black {
                        success!(id, &format!("W+{:.1}", white - black));
                    }
                } else {
                    error!(id, result.err().unwrap());
                }
            },
            Command::FinalStatusList(status) => {
                let board = self.history.last().unwrap();
                let result = self.ponder.service(|service, search_tree, p_state| {
                    let next_color = match board.last_played() {
                        Some(color) => color.opposite(),
                        _ => Color::Black,
                    };
                    let (finished, _rollout) = mcts::greedy_score(&service.lock(), &board, next_color);

                    (finished, Some(search_tree), p_state)
                });

                if let Ok(finished) = result {
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
                    error!(id, result.err().unwrap());
                }
            },
            Command::RegGenMove(color) => {
                let board = self.history.last().unwrap().clone();

                self.ponder = PonderService::new(board, color);
                self.generate_move(id, color, false);
            },
            Command::Undo => {
                if self.history.len() > 1 {
                    self.history.pop();

                    // update the ponder state with the new board position
                    let board = self.history.last().unwrap().clone();
                    let next_color = match board.last_played() {
                        Some(color) => color.opposite(),
                        None => Color::Black
                    };

                    self.ponder = PonderService::new(board, next_color);

                    success!(id, "");
                } else {
                    error!(id, "cannot undo");
                }
            },
            Command::TimeSettingsNone => {
                for &c in &[Color::Black, Color::White] {
                    self.time_settings[c as usize] = Box::new(time_settings::None::new());
                }

                success!(id, "");
            },
            Command::TimeSettingsAbsolute(main_time) => {
                for &c in &[Color::Black, Color::White] {
                    self.time_settings[c as usize] = Box::new(time_settings::Absolute::new(main_time));
                }

                success!(id, "");
            },
            Command::TimeSettingsByoYomi(main_time, byo_yomi_time, byo_yomi_stones) => {
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

                self.time_settings[c].time_left(main_time, byo_yomi_stones);
                success!(id, "");
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
        ponder: PonderService::new(Board::new(DEFAULT_KOMI), Color::Black),
        history: vec! [Board::new(DEFAULT_KOMI)],
        komi: DEFAULT_KOMI,
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
        assert_eq!(Gtp::parse_line("komi -7.5"), Some((None, Command::Komi(-7.5))));
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
        assert_eq!(Gtp::parse_line("1 time_settings 0 1 0"), Some((Some(1), Command::TimeSettingsNone)));
        assert_eq!(Gtp::parse_line("1 time_settings 30.2 0 0"), Some((Some(1), Command::TimeSettingsAbsolute(30.2))));
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
