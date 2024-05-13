use std::collections::HashMap;
use std::io::prelude::Write;
use std::io::{stdin, stdout};
use std::process::exit;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cozy_chess::util::{display_uci_move, parse_uci_move};
use cozy_chess::{Board, Color};
use frostburn::{Limits, LocalData, Search, SharedData};

type TokenIter<'a> = std::str::SplitAsciiWhitespace<'a>;
type CmdHandler = fn(&mut UciHandler, &mut TokenIter);

fn main() {
    // if std::env::args().nth(1).as_deref() == Some("bench") {
    //     bench();
    //     return;
    // }

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

struct UciHandler {
    position: Board,
    history: Vec<u64>,
    mv_format: MoveFormat,
    local_data: Arc<Mutex<LocalData>>,
    shared_data: Arc<RwLock<SharedData>>,
    threads: Vec<JoinHandle<()>>,
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
            shared_data: Arc::new(RwLock::new(SharedData::new())),
            threads: vec![],
        }
    }

    fn uci(&mut self, _: &mut TokenIter) {
        println!("id name Frostburn {}", env!("CARGO_PKG_VERSION_MAJOR"));
        println!("id author {}", env!("CARGO_PKG_AUTHORS"));
        println!("option name UCI_Chess960 type check default false");
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
            _ => {}
        }
    }

    fn position(&mut self, tokens: &mut TokenIter) {
        match tokens.next().unwrap() {
            "startpos" => self.position = Board::startpos(),
            "fen" => {
                let fen = tokens.take(6).fold(String::new(), |s, t| s + t + " ");
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
    }

    fn stop(&mut self, _: &mut TokenIter) {
        self.shared_data
            .read()
            .unwrap()
            .abort
            .store(true, Ordering::SeqCst);
        for thread in self.threads.drain(..) {
            thread.join().unwrap();
        }
    }

    fn go(&mut self, tokens: &mut TokenIter) {
        self.stop(tokens);

        let start = Instant::now();
        let white = self.position.side_to_move() == Color::White;

        let mut limits = Limits::default();

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
                "wtime" if white => limits.move_time = Some(Duration::from_millis(number(0)) / 40),
                "btime" if !white => limits.move_time = Some(Duration::from_millis(number(0)) / 40),
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
                    print!(
                        "info depth {d} score cp {s} nodes {n} time {t} nps {nps} pv",
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
