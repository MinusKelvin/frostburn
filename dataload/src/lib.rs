use std::ffi::OsStr;
use std::fs::{read_dir, File};
use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread::JoinHandle;

use cozy_chess::{Board, Color, Move, Piece, Square};
use datafmt::{DataReader, Game};
use rand::prelude::*;

fn filter(board: &Board, mv: Move, winner: Option<Color>) -> bool {
    if board.colors(!board.side_to_move()).has(mv.to) {
        return false;
    }
    thread_rng().gen_bool(0.25)
}

const BATCH_SIZE: usize = 1 << 14;

pub struct Loader {
    recv: Receiver<(Vec<[i64; 2]>, Vec<[i64; 2]>, Vec<f32>)>,
    handle: JoinHandle<()>,
}

#[no_mangle]
pub unsafe extern "C" fn create() -> *mut Loader {
    let mut streams = vec![];
    for f in read_dir("data/").unwrap() {
        let path = f.unwrap().path();
        if path.is_file() && path.extension() == Some(OsStr::new("fbdata")) {
            streams.push(data_stream(&path));
        }
    }
    assert!(!streams.is_empty(), "no data");

    let (send, recv) = sync_channel(2);
    let handle = std::thread::spawn(move || loop {
        let mut stm = Vec::with_capacity(BATCH_SIZE * 32);
        let mut nstm = Vec::with_capacity(BATCH_SIZE * 32);
        let mut targets = Vec::with_capacity(BATCH_SIZE);

        for i in 0..BATCH_SIZE {
            let (board, winner) = streams
                .choose_mut(&mut thread_rng())
                .unwrap()
                .next()
                .unwrap();

            for sq in board.occupied() {
                let color = board.color_on(sq).unwrap();
                let piece = board.piece_on(sq).unwrap();
                stm.push([i as i64, feature(board.side_to_move(), color, piece, sq)]);
                nstm.push([i as i64, feature(!board.side_to_move(), color, piece, sq)]);
            }

            targets.push(match winner {
                Some(c) if c == board.side_to_move() => 1.0,
                Some(_) => 0.0,
                None => 0.5,
            });
        }

        if send.send((stm, nstm, targets)).is_err() {
            break;
        };
    });

    Box::into_raw(Box::new(Loader { recv, handle }))
}

fn feature(stm: Color, color: Color, piece: Piece, sq: Square) -> i64 {
    let (color, sq) = match stm {
        Color::White => (color, sq),
        Color::Black => (!color, sq.flip_rank()),
    };

    let i = 0;
    let i = i * Color::NUM + color as usize;
    let i = i * Piece::NUM + piece as usize;
    let i = i * Square::NUM + sq as usize;

    i as i64
}

#[no_mangle]
pub unsafe extern "C" fn batch_size() -> u64 {
    BATCH_SIZE as u64
}

#[no_mangle]
pub unsafe extern "C" fn next_batch(
    loader: &mut Loader,
    stm_idx: &mut [[i64; 2]; BATCH_SIZE * 32],
    nstm_idx: &mut [[i64; 2]; BATCH_SIZE * 32],
    targets: &mut [f32; BATCH_SIZE],
) -> u64 {
    let (stm, nstm, t) = loader.recv.recv().unwrap();
    assert_eq!(stm.len(), nstm.len());
    assert_eq!(t.len(), BATCH_SIZE);

    targets.copy_from_slice(&t);
    stm_idx[..stm.len()].copy_from_slice(&stm);
    nstm_idx[..nstm.len()].copy_from_slice(&nstm);

    stm.len() as u64
}

#[no_mangle]
pub unsafe extern "C" fn destroy(loader: *mut Loader) {
    let loader = unsafe { *Box::from_raw(loader) };
    drop(loader.recv);
    loader.handle.join().unwrap();
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
