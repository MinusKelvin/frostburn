use std::collections::HashMap;
use std::io::prelude::Write;
use std::io::{stdin, stdout};
use std::process::exit;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cozy_chess::util::{display_uci_move, parse_uci_move};
use cozy_chess::{Board, Color};
use frostburn::{Accumulator, Limits, LocalData, Search, SharedData};

type TokenIter<'a> = std::str::SplitAsciiWhitespace<'a>;
type CmdHandler = fn(&mut UciHandler, &mut TokenIter);

fn main() {
    if std::env::args().nth(1).as_deref() == Some("bench") {
        bench();
        return;
    }

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
            shared_data: Arc::new(RwLock::new(SharedData::new(64))),
            threads: vec![],
        }
    }

    fn uci(&mut self, _: &mut TokenIter) {
        println!("id name Frostburn {}", env!("CARGO_PKG_VERSION_MAJOR"));
        println!("id author {}", env!("CARGO_PKG_AUTHORS"));
        println!("option name UCI_Chess960 type check default false");
        println!("option name Hash type spin min 1 max 1048576 default 64");
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
        self.shared_data.read().unwrap().abort();
        for thread in self.threads.drain(..) {
            thread.join().unwrap();
        }
    }

    fn eval(&mut self, _: &mut TokenIter) {
        let static_eval = Accumulator::new().infer(&self.position);
        println!("info string staticeval {static_eval}");
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

fn bench() {
    let mut shared = SharedData::new(8);
    let mut local = LocalData::new();

    let mut search_time = Duration::ZERO;
    let mut total_nodes = 0;
    for fen in BENCH_FENS {
        let root = fen.parse().unwrap();

        shared.prepare_for_search();
        let mut nodes = 0;
        let t = Instant::now();
        Search {
            root: &root,
            history: vec![],
            clock: &|| Duration::ZERO,
            info: &mut |info| nodes = info.nodes,
            data: &mut local,
            shared: &shared,
            limits: Limits {
                depth: Some(12),
                ..Default::default()
            },
        }
        .search();
        search_time += t.elapsed();

        total_nodes += nodes;
    }

    println!(
        "{total_nodes} nodes {} nps",
        ((total_nodes as f64) / search_time.as_secs_f64()) as u64
    );
}

const BENCH_FENS: &[&str] = &[
    "r4rk1/5pb1/3R2p1/p2Q1qBp/8/7P/1P3PP1/2R3K1 w - - 4 29",
    "r2qkbnr/ppp2p2/2npb3/4p1p1/2P1P2p/1PN1N3/P2PBPPP/R1BQK2R w KQq - 0 10",
    "3k4/8/4Q3/P2P4/8/5K1P/8/8 b - - 14 68",
    "1R6/2Pb1p2/6p1/2K2k2/7Q/8/5P1p/8 w - - 4 58",
    "r1bqk1r1/ppp1pp1p/2np2p1/7n/3PP2P/1PP1P3/PB1N2P1/R2QKB1R w KQq - 1 13",
    "3rr1k1/2p4p/1p1n2p1/nP1P1p1P/8/2B2PR1/P7/1K1R1B2 b - - 0 33",
    "5b2/2p2P1r/4k3/2P1P3/3B3p/7R/2K5/8 w - - 1 42",
    "8/p3rk1p/P1K1p1p1/3p4/8/r7/8/8 b - - 5 47",
    "rn1qk2r/pbp2ppp/3p1n2/1P2B3/1b6/2N2PP1/2PPP1BP/R2QK1NR w KQ - 0 10",
    "1nb1kbnr/rpp1qppp/8/pP1pp3/P6P/5P2/2PPP1P1/RNBQKBNR w KQk d6 0 6",
    "8/5pk1/6p1/2r5/4K1P1/8/8/8 b - g3 0 49",
    "r1bqkbr1/p2np2p/8/3p1p2/RP5Q/5N1P/2PP1PP1/2B1KB1R w K - 2 14",
    "r1b2rk1/1qnnbpp1/1p2p2p/p2pP3/3P4/P1NB1N2/1BQ2PPP/R5KR b - - 1 17",
    "5k2/5p2/2p2b1P/4rN2/1P1p1R2/3K2P1/P1P5/8 w - - 1 39",
    "8/8/1pbkp2p/2p1R3/2Pp1r2/pP1B3P/P7/4R1K1 b - - 1 37",
    "8/5pkp/5np1/4B3/3p1P1P/1P4P1/2n2PBK/8 b - - 0 37",
    "r2qkbnr/p1pp1pp1/4p2p/1P4P1/pn2P3/5N2/1BPP1P1P/RN1QK2R w KQkq - 1 10",
    "2krn1r1/p1p2qp1/Pp5p/4P2P/3p2Pb/2N1P3/1PPBB1K1/R2Q4 w - - 0 26",
    "r7/1p1b2k1/2np4/pB1p3p/P2P4/2PQPRp1/6P1/R4K1q w - - 3 30",
    "r2qk1r1/3pb1pp/p1b1pp2/2p1n3/Q3P3/2N3BP/PPP1BPP1/1R1R2K1 w q - 2 18",
    "2k2r2/1pp2Pq1/2n5/rN2p3/3pP3/PQ1P4/5R2/4R2K b - - 11 40",
    "6rk/2p4p/3p4/3Pp3/8/PB6/P7/3KQ1q1 b - - 4 36",
    "2r5/5pN1/3kpP1p/p2pn2P/Pp3R2/1P4r1/1KP1R1P1/8 b - - 2 45",
    "rn1qkbnr/2pppppp/b7/1p6/p4PP1/4PN1P/PPPP4/RNBQKB1R b KQkq - 2 5",
    "8/5pbk/7p/p1p2qp1/2P5/1P2B1P1/P6P/3RN1K1 b - - 2 32",
    "2kr3r/1pp2p1p/p1n5/5qPR/3N2P1/2P5/1P6/2KR4 w - - 0 25",
    "8/2k5/1p6/2r3p1/p2KB1n1/6P1/P4P2/1R3R2 b - - 1 43",
    "8/p7/5Np1/PP1kp3/5n1p/7P/5KP1/8 b - - 4 40",
    "r3kbnr/1pp1q1p1/2np2b1/pN2p2N/2P4p/P7/1P1PBPPP/R1BQ1RK1 b kq - 1 11",
    "r2qr1k1/7p/1p1p1ppP/pPpPnb2/3p4/P5R1/1BPPQPP1/2KR1B2 w - - 0 19",
    "rnbqkb1r/1p3p1p/p1pp1np1/4P3/4P3/N1PB3P/PP3PP1/R1BQK1NR b KQkq - 0 7",
    "4R3/6pk/5p1p/4p2P/8/4PKP1/2r2P2/8 b - - 3 41",
    "1r4k1/Q4ppp/8/8/4P3/8/K4PPP/1r3BR1 w - - 1 36",
    "r3k2r/pp2q3/2p5/5b1p/7R/P1P1PN2/PQ6/2K2B1R b - - 0 20",
    "2r3k1/1p5p/1q2p1pP/p3Pp2/P1PR1Q2/7P/6B1/6K1 w - - 4 33",
    "5rk1/p4pp1/1n2pq1p/2b5/3Np1P1/2Pn3P/P3QPB1/R2R2K1 b - - 1 24",
    "5rk1/5ppp/1p2p3/2n3P1/1NP2P2/4Q3/7P/5RK1 b - - 0 27",
    "rnb1k2r/pp3ppp/2p1pn2/8/P2PPB2/q1N3P1/2PQN1P1/1R2KB1R w Kkq - 2 13",
    "r4rk1/pp2pp1p/5qp1/5b2/1n1P4/2N1N2P/1P2BPP1/R2Q1RK1 b - - 4 17",
    "8/5kp1/p1p2p2/2P4p/RPb4P/6P1/3rrBP1/5RK1 w - - 14 39",
    "8/1p3N2/8/p1p3p1/3kP3/PP6/2P1K2r/5R2 w - - 3 43",
    "rnbqkbnr/p3pp1p/2p5/3p2B1/p7/2PPPN2/1P3PPP/RN1QKB1R b KQkq - 0 6",
    "8/3k4/5p2/2b2p2/2P1q3/3K4/8/8 w - - 0 47",
    "r3k1nr/1pp2pp1/p2p4/2PPp1qp/P3P3/5Q1P/1P3P2/2RK1B1R w kq - 0 16",
    "r1b1r1k1/pppn3p/5b2/3PP1p1/2N5/6P1/PPPB3P/R3K1NB w Q - 2 16",
    "8/5p2/1P1R2pk/7p/5P1P/4P3/1r6/6K1 w - - 5 49",
    "2krq1r1/1pp2p2/p3p3/2Npn3/6BR/2PPB1P1/PP3P2/R3K3 w - - 1 23",
    "4r1k1/6p1/2pn4/1pN5/4P1bP/1N4P1/2PP4/5R1K w - - 1 34",
    "r2qkbnr/ppp1p1pp/n4p2/3p4/3P1Pb1/N3P3/PPP3PP/R1BQKBNR w KQkq - 1 5",
    "6k1/2p2rp1/1p5p/rP6/4Pp1n/4NP1P/bB1RB1P1/1KR5 w - - 0 30",
    "r6k/ppp2B2/7p/7n/3n2P1/P1N5/1P5P/5RK1 b - - 0 22",
    "r1b1r1k1/2qnbppp/p1pp1n2/1p6/2PQP3/P1N1BN1P/1P3PP1/2KR1B1R b - - 0 13",
    "3R4/7P/6K1/1p4B1/4N3/8/1bk5/7r w - b6 0 61",
    "r1bq1rk1/pp3pp1/2pb1n2/3pn2p/8/1PN1PPP1/P1P2NBP/R1BQ2KR b - - 5 14",
    "3qk2r/2p1bpp1/2np4/1N2p2p/2P1P2n/3P1N1Q/PP3P1P/R1B2K1R w - h6 0 15",
    "2k5/pppb4/1b6/4p3/4N3/P2p4/1PP4P/2K3Q1 b - - 0 35",
    "r1r3k1/3bqpp1/1p2p2n/p2pP3/P2P1P2/1P1BNR1P/3Q3P/1R4K1 w - - 2 23",
    "r4rk1/1p2ppb1/2p1bnpp/q1P1N3/2P1PP2/2N3P1/PQ4BP/2R2R1K b - - 2 20",
    "8/p4p2/7p/1P6/3b1P2/2kB2P1/6K1/8 w - - 5 35",
    "2r2rk1/4N3/4q1p1/1Qp1p3/PnP3b1/4B1P1/1P1R1P2/4NRK1 b - - 0 32",
    "r1b1r1k1/p4pbp/2n2n2/q2p2Q1/2p5/P1N2P1P/1PP1N1P1/1RB1KB1R w K - 3 15",
    "2kr1b1r/1p1b1p2/pqn3pp/1N1Pp1P1/2P1p3/P2n4/4QN1P/R1B2RK1 b - - 0 22",
    "rnq1kb1r/pb1pn1p1/4p3/3p1pPp/2Q1P3/2N2P2/PPP3BP/R1B1K1NR w KQkq - 0 12",
    "rn1qkbnr/ppp1pppp/4b3/3p4/P1PP4/8/1P2PPPP/RNBQKBNR w KQkq - 0 5",
];
