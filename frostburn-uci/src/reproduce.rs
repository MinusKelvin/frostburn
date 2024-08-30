use std::io::prelude::Read;
use std::io::stdin;
use std::time::Instant;

use cozy_chess::util::parse_san_move;
use cozy_chess::{Board, Color};
use frostburn::{Limits, LocalData, Search, SharedData};

use crate::MoveFormat;

pub fn reproduce(side: Color, mb: usize) {
    let mut board = Board::startpos();

    for line in stdin()
        .lines()
        .map(Result::unwrap)
        .take_while(|l| !l.is_empty())
    {
        if line.starts_with("[FEN") {
            board = line.split('"').nth(1).unwrap().parse().unwrap();
        }
    }

    let mut shared = SharedData::new(mb);
    let mut local = LocalData::new();
    let mut history = vec![];

    let mut text = String::new();
    stdin().read_to_string(&mut text).unwrap();
    let mut tokens = text
        .split_ascii_whitespace()
        .inspect(|token| print!("{token} "));

    let mut last_mvnum = "";

    loop {
        if board.side_to_move() == Color::White {
            last_mvnum = match tokens.next() {
                Some(v) => v,
                None => break,
            };
        } else {
            print!("{last_mvnum}.. ");
        }

        let Ok(mv) = parse_san_move(&board, tokens.next().unwrap()) else {
            break;
        };
        let nodes = tokens
            .by_ref()
            .nth(3)
            .unwrap()
            .trim_matches(|c: char| !c.is_ascii_digit())
            .parse()
            .unwrap();

        println!();

        if board.side_to_move() == side {
            let start = Instant::now();
            shared.prepare_for_search();
            Search {
                root: &board,
                history: history.clone(),
                clock: &|| start.elapsed(),
                info: &mut |info| {
                    super::print_info(&board, MoveFormat::Chess960, &info);
                    if info.finished {
                        assert_eq!(info.nodes, nodes);
                        assert_eq!(info.pv[0], mv);
                    }
                },
                data: &mut local,
                shared: &shared,
                limits: Limits {
                    nodes: Some(nodes),
                    ..Default::default()
                },
            }
            .search();
        }

        history.push(board.hash());
        board.play(mv);
    }

    println!("Crashing search...");
    let start = Instant::now();
    shared.prepare_for_search();
    Search {
        root: &board,
        history: history.clone(),
        clock: &|| start.elapsed(),
        info: &mut |info| {
            super::print_info(&board, MoveFormat::Chess960, &info);
        },
        data: &mut local,
        shared: &shared,
        limits: Default::default(),
    }
    .search();
}
