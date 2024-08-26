use std::collections::HashMap;
use std::io::prelude::Write;
use std::io::{stdin, stdout};
use std::process::exit;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cozy_chess::util::{display_uci_move, parse_uci_move};
use cozy_chess::{Board, BoardBuilder, Color, Piece, Square};
use frostburn::{Accumulator, ClearTtBlock, Limits, LocalData, Search, SearchInfo, SharedData};

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
                    (tunable.max as f64 - tunable.min as f64) / 20.0
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
    shared_data: Arc<RwLock<(SearchConfig, SharedData)>>,
    threads: Vec<(SyncSender<Command>, JoinHandle<()>)>,
}

struct SearchConfig {
    position: Board,
    history: Vec<u64>,
    mv_format: MoveFormat,
    limits: Limits,

    start: Instant,
}

enum Command {
    Exit,
    ClearTt(ClearTtBlock),
    ResetData,
    Search,
    Rendezvous,
}

#[derive(Copy, Clone, Debug)]
enum MoveFormat {
    Standard,
    Chess960,
}

impl UciHandler {
    fn new() -> UciHandler {
        let mut this = UciHandler {
            shared_data: Arc::new(RwLock::new((
                SearchConfig {
                    position: Board::startpos(),
                    history: vec![],
                    start: Instant::now(),
                    mv_format: MoveFormat::Standard,
                    limits: Limits::default(),
                },
                SharedData::new(64),
            ))),
            threads: vec![],
        };
        this.set_option(&mut "name Threads value 1".split_ascii_whitespace());
        this
    }

    fn uci(&mut self, _: &mut TokenIter) {
        println!("id name Frostburn {}", env!("CARGO_PKG_VERSION_MAJOR"));
        println!("id author {}", env!("CARGO_PKG_AUTHORS"));
        println!("option name UCI_Chess960 type check default false");
        println!("option name Hash type spin min 1 max 1048576 default 64");
        println!("option name Threads type spin min 1 max 1024 default 1");
        println!("option name Weaken_Eval type spin min 0 max 10000 default 0");
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
        let mut guard = self.shared_data.write().unwrap();
        let (config, shared) = &mut *guard;
        match tokens.next().unwrap() {
            "UCI_Chess960" => {
                config.mv_format = match tokens.nth(1).unwrap() {
                    "true" => MoveFormat::Chess960,
                    "false" => MoveFormat::Standard,
                    _ => panic!("invalid value for UCI_Chess960"),
                }
            }
            "Hash" => {
                let mb = tokens.nth(1).unwrap().parse().unwrap();
                *shared = SharedData::new(mb);
            }
            "Threads" => {
                let num = tokens.nth(1).unwrap().parse().unwrap();
                for (send, t) in self.threads.drain(..) {
                    send.send(Command::Exit).unwrap();
                    t.join().unwrap();
                }
                for id in 0..num {
                    let (send, recv) = sync_channel(0);
                    let shared = self.shared_data.clone();
                    self.threads.push((
                        send,
                        std::thread::spawn(move || search_thread(shared, recv, id)),
                    ));
                }
            }
            "Weaken_Eval" => {
                config.limits.quantize_eval = tokens.nth(1).unwrap().parse::<i16>().unwrap() + 1;
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
        let mut guard = self.shared_data.write().unwrap();
        let (config, _) = &mut *guard;

        match tokens.next().unwrap() {
            "startpos" => config.position = Board::startpos(),
            "fen" => {
                let mut fen = tokens
                    .by_ref()
                    .take(4)
                    .fold(String::new(), |s, t| s + t + " ");
                fen += tokens.next_if(|&tok| tok != "moves").unwrap_or("0");
                fen += " ";
                fen += tokens.next_if(|&tok| tok != "moves").unwrap_or("1");
                config.position = fen.trim().parse().unwrap();
            }
            unknown => panic!("unknown position type {unknown}"),
        }

        config.history.clear();

        let _moves_token = tokens.next();

        while let Some(mv) = tokens.next() {
            let mv = match config.mv_format {
                MoveFormat::Standard => parse_uci_move(&config.position, mv).unwrap(),
                MoveFormat::Chess960 => mv.parse().unwrap(),
            };

            config.history.push(config.position.hash());
            config.position.play(mv);
        }
    }

    fn new_game(&mut self, _: &mut TokenIter) {
        for (send, _) in &self.threads {
            send.send(Command::ResetData).unwrap();
        }
        let blocks = self
            .shared_data
            .read()
            .unwrap()
            .1
            .get_clear_tt_blocks(self.threads.len());
        for ((send, _), block) in self.threads.iter().zip(blocks) {
            send.send(Command::ClearTt(block)).unwrap();
        }
        for (send, _) in &self.threads {
            send.send(Command::Rendezvous).unwrap();
        }
    }

    fn stop(&mut self, _: &mut TokenIter) {
        self.shared_data.read().unwrap().1.abort();
        for (send, _) in &self.threads {
            send.send(Command::Rendezvous).unwrap();
        }
    }

    fn eval(&mut self, _: &mut TokenIter) {
        let mut acc = Accumulator::new();
        let guard = self.shared_data.read().unwrap();
        let config = &guard.0;
        let static_eval = acc.infer(&config.position);
        let mut others = [None; 64];
        let remove_pieces = config.position.occupied() - config.position.pieces(Piece::King);
        for sq in remove_pieces {
            let mut board = BoardBuilder::from_board(&config.position);
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
                        1 if config.position.occupied().has(sq) => {
                            print!(
                                "|  {} {}  ",
                                config.position.color_on(sq).unwrap(),
                                config.position.piece_on(sq).unwrap(),
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
        let start = Instant::now();

        let mut guard = self.shared_data.write().unwrap();
        let (config, shared) = &mut *guard;

        let white = config.position.side_to_move() == Color::White;

        config.limits.unbounded();
        config.start = start;

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
                "movetime" => config.limits.move_time = Some(Duration::from_millis(number(0))),
                "depth" => config.limits.depth = Some(number(0) as i16),
                "nodes" => config.limits.nodes = Some(number(0)),
                "minnodes" => config.limits.min_nodes = Some(number(0)),
                "wtime" if white => config.limits.clock = Some(Duration::from_millis(number(0))),
                "btime" if !white => config.limits.clock = Some(Duration::from_millis(number(0))),
                "winc" if white => config.limits.increment = Duration::from_millis(number(0)),
                "binc" if !white => config.limits.increment = Duration::from_millis(number(0)),
                _ => {}
            }
        }

        shared.prepare_for_search();

        drop(guard);
        for (send, _) in &self.threads {
            send.send(Command::Search).unwrap();
        }
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

fn search_thread(
    shared_data: Arc<RwLock<(SearchConfig, SharedData)>>,
    command: Receiver<Command>,
    id: usize,
) {
    let mut local_data = LocalData::new();
    loop {
        match command.recv().unwrap_or(Command::Exit) {
            Command::Exit => return,
            Command::ClearTt(range) => {
                shared_data.read().unwrap().1.clear_tt_block(range);
                continue;
            }
            Command::ResetData => {
                local_data = LocalData::new();
                continue;
            }
            Command::Rendezvous => continue,
            Command::Search => {}
        }

        let guard = shared_data.read().unwrap();
        let (config, shared) = &*guard;

        let mut limits = config.limits;

        let info: &mut dyn FnMut(SearchInfo) = match id {
            0 => &mut |info| {
                print_info(&config.position, config.mv_format, &info);

                if info.finished {
                    match config.mv_format {
                        MoveFormat::Standard => println!(
                            "bestmove {}",
                            display_uci_move(&config.position, info.pv[0])
                        ),
                        MoveFormat::Chess960 => println!("bestmove {}", info.pv[0]),
                    }
                }

                stdout().flush().unwrap();
            },
            _ => {
                limits.unbounded();
                &mut |_| {}
            }
        };

        let clock: &dyn Fn() -> Duration = match id {
            0 => &|| config.start.elapsed(),
            _ => &|| Duration::ZERO,
        };

        Search {
            root: &config.position,
            history: config.history.clone(),
            clock,
            info,
            data: &mut local_data,
            shared,
            limits,
        }
        .search()
    }
}
