use arrayvec::ArrayVec;
use cozy_chess::{BitBoard, Board, Color, Piece, Square};

use crate::Eval;

#[derive(Clone)]
pub struct Accumulator {
    enabled: [[BitBoard; 6]; 2],
    white: [i16; 256],
    black: [i16; 256],
}

#[repr(C)]
struct Linear<T, const IN: usize, const OUT: usize> {
    w: [[T; OUT]; IN],
    bias: [T; OUT],
}

#[repr(C)]
struct Network {
    ft: Linear<i16, 768, 256>,
    l1: Linear<i16, 512, 1>,
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            enabled: [[BitBoard::EMPTY; 6]; 2],
            white: NETWORK.ft.bias,
            black: NETWORK.ft.bias,
        }
    }

    pub fn infer(&mut self, board: &Board) -> Eval {
        let mut white_adds = ArrayVec::<_, 32>::new();
        let mut white_rms = ArrayVec::<_, 32>::new();
        let mut black_adds = ArrayVec::<_, 32>::new();
        let mut black_rms = ArrayVec::<_, 32>::new();
        for color in Color::ALL {
            for piece in Piece::ALL {
                let enabled = &mut self.enabled[color as usize][piece as usize];
                let feats = board.colored_pieces(color, piece);
                let removed = *enabled - feats;
                let added = feats - *enabled;
                *enabled = feats;

                for sq in removed {
                    white_rms.push(&NETWORK.ft.w[feature(Color::White, color, piece, sq)]);
                    black_rms.push(&NETWORK.ft.w[feature(Color::Black, color, piece, sq)]);
                }
                for sq in added {
                    white_adds.push(&NETWORK.ft.w[feature(Color::White, color, piece, sq)]);
                    black_adds.push(&NETWORK.ft.w[feature(Color::Black, color, piece, sq)]);
                }
            }
        }

        update(&mut self.white, &white_adds, &white_rms);
        update(&mut self.black, &black_adds, &black_rms);

        let mut activated = [0; 512];
        let (left, right) = activated.split_at_mut(256);
        let left = <&mut [_; 256]>::try_from(left).unwrap();
        let right = <&mut [_; 256]>::try_from(right).unwrap();

        match board.side_to_move() {
            Color::White => {
                *left = crelu(&self.white);
                *right = crelu(&self.black);
            }
            Color::Black => {
                *left = crelu(&self.black);
                *right = crelu(&self.white);
            }
        }

        let mut result = NETWORK.l1.bias[0] as i32;

        for i in 0..512 {
            result += activated[i] as i32 * NETWORK.l1.w[i][0] as i32;
        }

        Eval::cp((result / 128).clamp(-29_000, 29_000) as i16)
    }
}

fn update<const N: usize>(acc: &mut [i16; N], adds: &[&[i16; N]], rms: &[&[i16; N]]) {
    for add in adds {
        for i in 0..N {
            acc[i] += add[i];
        }
    }
    for rm in rms {
        for i in 0..N {
            acc[i] -= rm[i];
        }
    }
}

fn crelu<const N: usize>(a: &[i16; N]) -> [i16; N] {
    let mut result = [0; N];
    for i in 0..N {
        result[i] = a[i].clamp(0, 255);
    }
    result
}

fn feature(stm: Color, color: Color, piece: Piece, sq: Square) -> usize {
    let (color, sq) = match stm {
        Color::White => (color, sq),
        Color::Black => (!color, sq.flip_rank()),
    };

    let i = 0;
    let i = i * Color::NUM + color as usize;
    let i = i * Piece::NUM + piece as usize;
    let i = i * Square::NUM + sq as usize;

    i
}

#[cfg(target_endian = "little")]
static NETWORK: Network = unsafe { core::mem::transmute(*include_bytes!(env!("NNUE_PATH"))) };

#[cfg(not(target_endian = "little"))]
compile_error!("Non-little endian targets are not currently supported.");
