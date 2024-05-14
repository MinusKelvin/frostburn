use std::fs::File;
use std::io::prelude::Write;
use std::io::stdout;
use std::num::ParseIntError;
use std::path::PathBuf;
use std::process::exit;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use cozy_chess::{Board, GameStatus};
use datafmt::{DataWriter, Game};
use frostburn::{Limits, LocalData, Search, SharedData};
use rand::prelude::*;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
struct Options {
    output: PathBuf,
    #[structopt(long, short, default_value = "100k", parse(try_from_str = human_parse))]
    games: u64,
    #[structopt(long, short, default_value = "5k", parse(try_from_str = human_parse))]
    nodes: u64,
    #[structopt(long)]
    opp_nodes: Option<u64>,
}

fn main() {
    let options = Options::from_args();

    if options.output.exists() {
        eprintln!("Refusing to overwrite existing file.");
        exit(1);
    }

    let writer = Mutex::new(DataWriter::new(File::create(options.output).unwrap()).unwrap());
    let start = Instant::now();
    let concurrency = std::thread::available_parallelism().unwrap().get();

    std::thread::scope(|s| {
        for _ in 0..concurrency {
            s.spawn(|| loop {
                let game = play_game(options.nodes, options.opp_nodes.unwrap_or(options.nodes));

                let mut guard = writer.lock().unwrap();

                guard.write_game(&game).unwrap();
                let header = guard.header();

                let completion = header.count() as f64 / options.games as f64;
                let spent = start.elapsed().as_secs_f64();
                let time = match header.count() == options.games {
                    false => (spent / completion - spent) as i64,
                    true => spent as i64,
                };
                let secs = time % 60;
                let mins = time / 60 % 60;
                let hours = time / 3600 % 24;
                let days = time / 86400;

                print!(
                    "\r{:>6.2}% {:>10} positions {:>5}/sec",
                    completion * 100.0,
                    header.nominal_positions,
                    (header.nominal_positions as f64 / spent) as u64
                );

                match header.count() == options.games {
                    false => print!("   ETA "),
                    true => print!("  Took "),
                }

                print!("{days:>2}d {hours:>2}h {mins:>2}m {secs:>2}s");

                if let Some((elo, conf)) = header.elo() {
                    print!("    Elo: {:>6.2} +- {:<5.2}  ", elo, conf);
                }

                stdout().flush().unwrap();

                if header.count() > options.games - concurrency as u64 {
                    break;
                }
            });
        }
    });
    println!();

    writer.into_inner().unwrap().finish().unwrap();
}

fn play_game(a_nodes: u64, b_nodes: u64) -> Game {
    let (mut game, mut board) = pick_startpos();

    let mut limits = [Limits::default(); 2];
    limits[0].min_nodes = Some(a_nodes);
    limits[1].min_nodes = Some(b_nodes);

    let mut local = [LocalData::new(), LocalData::new()];
    let mut shared = [SharedData::new(), SharedData::new()];

    let mut history = vec![];

    loop {
        let idx = board.side_to_move() as usize;
        let mut mv = None;
        shared[idx].prepare_for_search();
        Search {
            root: &board,
            history: history.clone(),
            clock: &|| Duration::ZERO,
            info: &mut |info| {
                if info.finished {
                    mv = Some(info.pv[0]);
                }
            },
            data: &mut local[idx],
            shared: &mut shared[idx],
            limits: limits[idx],
        }
        .search();

        let mv = mv.unwrap_or_else(|| panic!("no move in position {board}"));

        history.push(board.hash());
        board.play(mv);
        game.moves.push(mv);

        match board.status() {
            GameStatus::Won => {
                game.winner = Some(!board.side_to_move());
                return game;
            }
            GameStatus::Drawn => {
                game.winner = None;
                return game;
            }
            GameStatus::Ongoing => {
                if history.iter().filter(|&&h| h == board.hash()).count() == 3 {
                    game.winner = None;
                    return game;
                }
            }
        }
    }
}

fn pick_startpos() -> (Game, Board) {
    let mut moves = vec![];
    let mut moves_to_pick = vec![];
    'retry: loop {
        let white_scharnagl = thread_rng().gen_range(0..960);
        let black_scharnagl = thread_rng().gen_range(0..960);
        let color_flipped = thread_rng().gen_bool(0.5);
        let fake_moves = thread_rng().gen_range(8..10);

        let mut board =
            Board::double_chess960_startpos(white_scharnagl as u32, black_scharnagl as u32);
        if color_flipped {
            board = board.null_move().unwrap();
        }

        moves.clear();
        for _ in 0..fake_moves {
            moves_to_pick.clear();
            board.generate_moves(|mvs| {
                moves_to_pick.extend(mvs);
                false
            });
            let Some(&mv) = moves_to_pick.choose(&mut thread_rng()) else {
                continue 'retry;
            };
            board.play_unchecked(mv);
            moves.push(mv);
        }

        if board.status() != GameStatus::Ongoing {
            continue 'retry;
        }

        let game = Game {
            white_scharnagl,
            black_scharnagl,
            fake_moves,
            color_flipped,
            winner: None,
            moves,
        };

        return (game, board);
    }
}

fn human_parse(s: &str) -> Result<u64, ParseIntError> {
    let without_underscores = s.replace("_", "");

    if let Some(number) = without_underscores.strip_suffix("k") {
        Ok(number.parse::<u64>()? * 1_000)
    } else if let Some(number) = without_underscores.strip_suffix("M") {
        Ok(number.parse::<u64>()? * 1_000_000)
    } else if let Some(number) = without_underscores.strip_suffix("G") {
        Ok(number.parse::<u64>()? * 1_000_000_000)
    } else {
        without_underscores.parse()
    }
}
