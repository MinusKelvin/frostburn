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
    if mv.promotion.is_some() {
        return false;
    }
    thread_rng().gen_bool(0.1)
}

const BATCH_SIZE: usize = 1 << 14;
const SIMUL_BATCHES: usize = 128;

const BLACK_FLIP: usize = 0b1_111_000;
const MIRROR_FLIP: usize = 0b0_000_111;

pub struct Loader {
    batches: usize,
    recv: Receiver<Batch>,
    handles: Vec<JoinHandle<()>>,
}

struct Batch {
    stm: Vec<[i64; 2]>,
    nstm: Vec<[i64; 2]>,
    targets: Vec<f32>,
}

#[no_mangle]
pub unsafe extern "C" fn create() -> *mut Loader {
    let mut streams = vec![];
    let mut handles = vec![];
    for f in read_dir("data/").unwrap() {
        let path = f.unwrap().path();
        if path.is_file() && path.extension() == Some(OsStr::new("fbdata")) {
            let (send, recv) = sync_channel(4);
            let mut stream = data_stream(&path);
            streams.push(recv);
            handles.push(std::thread::spawn(move || loop {
                let data = [(); SIMUL_BATCHES].map(|_| stream.next().unwrap());
                if send.send(data).is_err() {
                    return;
                }
            }));
        }
    }
    assert!(!streams.is_empty(), "no data");

    let (send, recv) = sync_channel(SIMUL_BATCHES);
    handles.push(std::thread::spawn(move || loop {
        let mut batches = [(); SIMUL_BATCHES].map(|_| Batch::default());

        for i in 0..BATCH_SIZE {
            let datas = streams
                .choose_mut(&mut thread_rng())
                .unwrap()
                .recv()
                .unwrap();

            for ((board, winner), batch) in datas.into_iter().zip(&mut batches) {
                for sq in board.occupied() {
                    let color = board.color_on(sq).unwrap();
                    let piece = board.piece_on(sq).unwrap();

                    let mut stm_feature = feature(color, piece, sq);
                    let mut nstm_feature = stm_feature ^ BLACK_FLIP;

                    if board.side_to_move() == Color::Black {
                        std::mem::swap(&mut stm_feature, &mut nstm_feature);
                    }

                    if board.king(board.side_to_move()).file() >= cozy_chess::File::E {
                        stm_feature ^= MIRROR_FLIP;
                    }
                    if board.king(!board.side_to_move()).file() >= cozy_chess::File::E {
                        nstm_feature ^= MIRROR_FLIP;
                    }

                    batch.stm.push([i as i64, stm_feature as i64]);
                    batch.nstm.push([i as i64, nstm_feature as i64]);
                }

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
    }));

    Box::into_raw(Box::new(Loader {
        batches: 0,
        recv,
        handles,
    }))
}

fn feature(color: Color, piece: Piece, sq: Square) -> usize {
    let i = 0;
    let i = i * Piece::NUM + piece as usize;
    let i = i * Color::NUM + color as usize;
    let i = i * Square::NUM + sq as usize;

    i
}

#[no_mangle]
pub unsafe extern "C" fn batch_size() -> u64 {
    BATCH_SIZE as u64
}

#[no_mangle]
pub unsafe extern "C" fn next_batch(
    loader: &mut Loader,
    stm: &mut [[i64; 2]; BATCH_SIZE * 32],
    nstm: &mut [[i64; 2]; BATCH_SIZE * 32],
    targets: &mut [f32; BATCH_SIZE],
) -> u64 {
    loader.batches += 1;
    // let t = std::time::Instant::now();
    let batch = match loader.recv.try_recv() {
        Ok(v) => v,
        Err(_) => {
            let v = loader.recv.recv().unwrap();
            // eprintln!("batch {} delayed: {:?}", loader.batches, t.elapsed());
            v
        }
    };
    assert_eq!(batch.stm.len(), batch.nstm.len());
    assert_eq!(batch.targets.len(), BATCH_SIZE);

    targets.copy_from_slice(&batch.targets);
    stm[..batch.stm.len()].copy_from_slice(&batch.stm);
    nstm[..batch.nstm.len()].copy_from_slice(&batch.nstm);

    batch.stm.len() as u64
}

#[no_mangle]
pub unsafe extern "C" fn destroy(loader: *mut Loader) {
    let loader = unsafe { *Box::from_raw(loader) };
    drop(loader.recv);
    for handle in loader.handles {
        handle.join().unwrap();
    }
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
