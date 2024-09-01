use std::ffi::OsStr;
use std::fs::{read_dir, File};
use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver};

use cozy_chess::{Board, Color, Move, Piece, Square};
use datafmt::{DataReader, Game};
use rand::prelude::*;

fn filter(board: &Board, mv: Move, winner: Option<Color>) -> bool {
    if board.colors(!board.side_to_move()).has(mv.to) {
        return false;
    }
    thread_rng().gen_bool(0.1)
}

pub const BATCH_SIZE: usize = 1 << 14;
const SIMUL_BATCHES: usize = 128;

pub struct Batch {
    pub stm: Vec<[i64; 32]>,
    pub nstm: Vec<[i64; 32]>,
    pub targets: Vec<f32>,
}

pub fn spawn_data_loader() -> Receiver<Batch> {
    let mut streams = vec![];
    for f in read_dir("data/").unwrap() {
        let path = f.unwrap().path();
        if path.is_file() && path.extension() == Some(OsStr::new("fbdata")) {
            let (send, recv) = sync_channel(4);
            let mut stream = data_stream(&path);
            streams.push(recv);
            std::thread::spawn(move || loop {
                let data = [(); SIMUL_BATCHES].map(|_| stream.next().unwrap());
                if send.send(data).is_err() {
                    return;
                }
            });
        }
    }
    assert!(!streams.is_empty(), "no data");

    let (send, recv) = sync_channel(SIMUL_BATCHES);
    std::thread::spawn(move || loop {
        let mut batches = [(); SIMUL_BATCHES].map(|_| Batch::default());

        for i in 0..BATCH_SIZE {
            let datas = streams
                .choose_mut(&mut thread_rng())
                .unwrap()
                .recv()
                .unwrap();

            for ((board, winner), batch) in datas.into_iter().zip(&mut batches) {
                let mut stm = [-1; 32];
                let mut nstm = [-1; 32];
                for (i, sq) in board.occupied().iter().enumerate() {
                    let color = board.color_on(sq).unwrap();
                    let piece = board.piece_on(sq).unwrap();
                    stm[i] = feature(board.side_to_move(), color, piece, sq);
                    nstm[i] = feature(!board.side_to_move(), color, piece, sq);
                }

                batch.stm.push(stm);
                batch.nstm.push(nstm);
                batch.targets.push(match winner {
                    Some(c) if c == board.side_to_move() => 1.0,
                    Some(_) => 0.0,
                    None => 0.5,
                });
            }
        }

        batches.shuffle(&mut thread_rng());
        for batch in batches {
            if send.send(batch).is_err() {
                return;
            };
        }
    });

    recv
}

fn feature(stm: Color, color: Color, piece: Piece, sq: Square) -> i64 {
    let (color, sq) = match stm {
        Color::White => (color, sq),
        Color::Black => (!color, sq.flip_rank()),
    };

    let i = 0;
    let i = i * Piece::NUM + piece as usize;
    let i = i * Color::NUM + color as usize;
    let i = i * Square::NUM + sq as usize;

    i as i64
}

fn data_stream(file: &Path) -> impl Iterator<Item = (Board, Option<Color>)> {
    GameStream::new(file)
        .flat_map(|game| {
            let mut board = Board::double_chess960_startpos(
                game.white_scharnagl as u32,
                game.black_scharnagl as u32,
            );
            if game.color_flipped {
                board = board.null_move().unwrap();
            }

            game.moves
                .into_iter()
                .scan(board, move |board, mv| {
                    let pos = board.clone();
                    board.play(mv);
                    Some((pos, mv, game.winner))
                })
                .skip(game.fake_moves as usize)
        })
        .filter(|&(ref board, mv, winner)| filter(board, mv, winner))
        .map(|(board, _, winner)| (board, winner))
}

struct GameStream {
    from: DataReader,
}

impl GameStream {
    fn new(path: &Path) -> Self {
        GameStream {
            from: DataReader::new(File::open(path).unwrap()).unwrap(),
        }
    }
}

impl Iterator for GameStream {
    type Item = Game;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.from.read_game().unwrap() {
                Some(v) => return Some(v),
                None => self.from.reset().unwrap(),
            }
        }
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self {
            stm: Vec::with_capacity(BATCH_SIZE * 32),
            nstm: Vec::with_capacity(BATCH_SIZE * 32),
            targets: Vec::with_capacity(BATCH_SIZE),
        }
    }
}
