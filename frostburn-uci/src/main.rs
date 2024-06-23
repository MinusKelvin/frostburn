use std::collections::HashMap;
use std::io::prelude::Write;
use std::io::{stdin, stdout};
use std::process::exit;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cozy_chess::util::{display_uci_move, parse_uci_move};
use cozy_chess::{Board, BoardBuilder, Color, Piece, Square};
use frostburn::{Accumulator, Limits, LocalData, Search, SearchInfo, SharedData};

mod bench;
mod reproduce;

type TokenIter<'a> = std::str::SplitAsciiWhitespace<'a>;
type CmdHandler = fn(&mut UciHandler, &mut TokenIter);

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("bench") => bench::bench(),
        Some("reproduce") => {
            let err = || -> ! {
                eprintln!("usage: reproduce <white|black> <hash>");
                eprintln!("supply openbench pgn on stdin");
                exit(1);
            };
            let side = match args.next().as_deref() {
                Some("white") => Color::White,
                Some("black") => Color::Black,
                _ => err(),
            };
            let mb = args
                .next()
                .as_deref()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| err());
            reproduce::reproduce(side, mb);
        }
        #[cfg(feature = "tunable")]
        Some("spsa") => {
            for tunable in frostburn::TUNABLES {
                println!(
                    "{}, int, {}, {}, {}, {}, 0.002",
                    tunable.name,
                    tunable.default,
                    tunable.min,
                    tunable.max,
                    (tunable.default as f64 * 0.2).abs().max(0.2)
                );
            }
        }
        Some(c) => {
            eprintln!("Unrecognized command: `{c}`");
            exit(1);
        }
        None => {
            let mut cmds = HashMap::new();
            cmds.insert("uci", UciHandler::uci as CmdHandler);
            cmds.insert("isready", UciHandler::is_ready);
            cmds.insert("quit", UciHandler::quit);
            cmds.insert("position", UciHandler::position);
            cmds.insert("go", UciHandler::go);
            cmds.insert("debug", |_, _| {});
            cmds.insert("setoption", UciHandler::set_option);
            cmds.insert("ucinewgame", UciHandler::new_game);
            cmds.insert("stop", UciHandler::stop);
            cmds.insert("eval", UciHandler::eval);

            let mut uci = UciHandler::new();
            let mut buf = String::new();
            loop {
                buf.clear();
                if stdin().read_line(&mut buf).unwrap() == 0 {
                    exit(0);
                }
                let mut tokens = buf.split_ascii_whitespace();
                let Some(cmd) = tokens.next() else { continue };
                let Some(&handler) = cmds.get(cmd) else {
                    panic!("Unknown command {cmd}")
                };
                handler(&mut uci, &mut tokens);
            }
        }
    }
}

struct UciHandler {
    position: Board,
    history: Vec<u64>,
    mv_format: MoveFormat,
    local_data: Arc<Mutex<LocalData>>,
    shared_data: Arc<RwLock<SharedData>>,
    threads: Vec<JoinHandle<()>>,
    randomize_eval: i16,
}

#[derive(Copy, Clone, Debug)]
enum MoveFormat {
    Standard,
    Chess960,
}

impl UciHandler {
    fn new() -> UciHandler {
        UciHandler {
            position: Board::startpos(),
            history: vec![],
            mv_format: MoveFormat::Standard,
            local_data: Arc::new(Mutex::new(LocalData::new())),
            shared_data: Arc::new(RwLock::new(SharedData::new(64))),
            threads: vec![],
            randomize_eval: 0,
        }
    }

    fn uci(&mut self, _: &mut TokenIter) {
        println!("id name Frostburn {}", env!("CARGO_PKG_VERSION_MAJOR"));
        println!("id author {}", env!("CARGO_PKG_AUTHORS"));
        println!("option name UCI_Chess960 type check default false");
        println!("option name Hash type spin min 1 max 1048576 default 64");
        println!("option name Weaken_EvalNoise type spin min 0 max 10000 default 0");
        #[cfg(feature = "tunable")]
        for tunable in frostburn::TUNABLES {
            println!(
                "option name {} type spin min {} max {} default {}",
                tunable.name, tunable.min, tunable.max, tunable.default
            );
        }
        println!("uciok");
    }

    fn is_ready(&mut self, _: &mut TokenIter) {
        println!("readyok");
    }

    fn quit(&mut self, _: &mut TokenIter) {
        exit(0)
    }

    fn set_option(&mut self, tokens: &mut TokenIter) {
        let _name_token = tokens.next();
        match tokens.next().unwrap() {
            "UCI_Chess960" => {
                self.mv_format = match tokens.nth(1).unwrap() {
                    "true" => MoveFormat::Chess960,
                    "false" => MoveFormat::Standard,
                    _ => panic!("invalid value for UCI_Chess960"),
                }
            }
            "Hash" => {
                let mb = tokens.nth(1).unwrap().parse().unwrap();
                *self.shared_data.write().unwrap() = SharedData::new(mb);
            }
            "Weaken_EvalNoise" => {
                self.randomize_eval = tokens.nth(1).unwrap().parse().unwrap();
            }
            #[cfg(feature = "tunable")]
            param => {
                let v = tokens.nth(1).unwrap().parse().unwrap();
                for tunable in frostburn::TUNABLES {
                    if param == tunable.name {
                        tunable.atomic.store(v, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            }
            _ => {}
        }
    }

    fn position(&mut self, tokens: &mut TokenIter) {
        let mut tokens = tokens.peekable();

        match tokens.next().unwrap() {
            "startpos" => self.position = Board::startpos(),
            "fen" => {
                let mut fen = tokens
                    .by_ref()
                    .take(4)
                    .fold(String::new(), |s, t| s + t + " ");
                fen += tokens.next_if(|&tok| tok != "moves").unwrap_or("0");
                fen += " ";
                fen += tokens.next_if(|&tok| tok != "moves").unwrap_or("1");
                self.position = fen.trim().parse().unwrap();
            }
            unknown => panic!("unknown position type {unknown}"),
        }

        self.history.clear();

        let _moves_token = tokens.next();

        while let Some(mv) = tokens.next() {
            let mv = match self.mv_format {
                MoveFormat::Standard => parse_uci_move(&self.position, mv).unwrap(),
                MoveFormat::Chess960 => mv.parse().unwrap(),
            };

            self.history.push(self.position.hash());
            self.position.play(mv);
        }
    }

    fn new_game(&mut self, _: &mut TokenIter) {
        self.local_data = Arc::new(Mutex::new(LocalData::new()));
        self.shared_data.write().unwrap().clear_tt();
    }

    fn stop(&mut self, _: &mut TokenIter) {
        self.shared_data.read().unwrap().abort();
        for thread in self.threads.drain(..) {
            thread.join().unwrap();
        }
    }

    fn eval(&mut self, _: &mut TokenIter) {
        let mut acc = Accumulator::new();
        let static_eval = acc.infer(&self.position);
        let mut others = [None; 64];
        let remove_pieces = self.position.occupied() - self.position.pieces(Piece::King);
        for sq in remove_pieces {
            let mut board = BoardBuilder::from_board(&self.position);
            board.board[sq as usize] = None;
            board.castle_rights[0].short = None;
            board.castle_rights[0].long = None;
            board.castle_rights[1].short = None;
            board.castle_rights[1].long = None;
            if let Ok(board) = board.build() {
                others[sq as usize] = Some(acc.infer(&board));
            }
        }

        for rank in (0..8).rev() {
            for l in 0..3 {
                for file in 0..8 {
                    let idx = rank * 8 + file;
                    let sq = Square::index(idx);

                    match l {
                        0 => print!("+-------"),
                        1 if self.position.occupied().has(sq) => {
                            print!(
                                "|  {} {}  ",
                                self.position.color_on(sq).unwrap(),
                                self.position.piece_on(sq).unwrap()
                            );
                        }
                        1 => print!("|       "),
                        2 => match others[idx] {
                            Some(without) => print!("| {:^+5} ", static_eval - without),
                            None => print!("|       "),
                        },
                        _ => unreachable!(),
                    }
                }
                match l {
                    0 => println!("+"),
                    1 | 2 => println!("|"),
                    _ => unreachable!(),
                }
            }
        }
        for _ in 0..8 {
            print!("+-------");
        }
        println!("+");

        println!("{static_eval}");
    }

    fn go(&mut self, tokens: &mut TokenIter) {
        self.stop(tokens);

        let start = Instant::now();
        let white = self.position.side_to_move() == Color::White;

        let mut limits = Limits::default();
        limits.randomize_eval = self.randomize_eval;

        while let Some(limit_verb) = tokens.next() {
            let mut number = |if_negative: u64| {
                tokens
                    .next()
                    .unwrap()
                    .parse::<i64>()
                    .unwrap()
                    .try_into()
                    .unwrap_or(if_negative)
            };
            match limit_verb {
                "movetime" => limits.move_time = Some(Duration::from_millis(number(0))),
                "depth" => limits.depth = Some(number(0) as i16),
                "nodes" => limits.nodes = Some(number(0)),
                "minnodes" => limits.min_nodes = Some(number(0)),
                "wtime" if white => limits.clock = Some(Duration::from_millis(number(0))),
                "btime" if !white => limits.clock = Some(Duration::from_millis(number(0))),
                _ => {}
            }
        }

        self.shared_data.write().unwrap().prepare_for_search();

        let root = self.position.clone();
        let history = self.history.clone();
        let local = self.local_data.clone();
        let shared = self.shared_data.clone();
        let mv_format = self.mv_format;
        self.threads.push(std::thread::spawn(move || {
            Search {
                root: &root,
                history,
                clock: &|| start.elapsed(),
                info: &mut |info| {
                    print_info(&root, mv_format, &info);

                    if info.finished {
                        match mv_format {
                            MoveFormat::Standard => {
                                println!("bestmove {}", display_uci_move(&root, info.pv[0]))
                            }
                            MoveFormat::Chess960 => println!("bestmove {}", info.pv[0]),
                        }
                    }

                    stdout().flush().unwrap();
                },
                data: &mut local.lock().unwrap(),
                shared: &shared.read().unwrap(),
                limits,
            }
            .search()
        }));
    }
}

fn print_info(root: &Board, mv_format: MoveFormat, info: &SearchInfo) {
    print!(
        "info depth {d} score {s} nodes {n} time {t} nps {nps} pv",
        d = info.depth,
        s = info.score,
        n = info.nodes,
        t = info.time.as_millis(),
        nps = (info.nodes as f64 / info.time.as_secs_f64()) as u64,
    );

    let mut board = root.clone();
    for &mv in info.pv {
        match mv_format {
            MoveFormat::Standard => print!(" {}", display_uci_move(&board, mv)),
            MoveFormat::Chess960 => print!(" {}", mv),
        }
        board.play_unchecked(mv);
    }

    println!();
}
