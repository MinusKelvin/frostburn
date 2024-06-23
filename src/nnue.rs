use arrayvec::ArrayVec;
use cozy_chess::{BitBoard, Board, Color, Piece, Square};

use crate::Eval;

#[cfg(target_arch = "x86_64")]
mod avx2;

mod scalar;

#[derive(Clone)]
pub struct Accumulator {
    enabled: [[BitBoard; 6]; 2],
    white: [i16; 512],
    black: [i16; 512],
}

#[repr(C)]
struct FeatureTransformer<const IN: usize, const OUT: usize> {
    w: [[i16; OUT]; IN],
    bias: [i16; OUT],
}

#[repr(C)]
struct Linear<const IN: usize, const OUT: usize> {
    w: [[i16; IN]; OUT],
    bias: [i32; OUT],
}

#[repr(C)]
struct Network {
    ft: FeatureTransformer<768, 512>,
    l1: Linear<1024, 1>,
}

#[derive(Default)]
struct Updates<'a, const N: usize = 512> {
    white_adds: ArrayVec<&'a [i16; N], 32>,
    white_rms: ArrayVec<&'a [i16; N], 32>,
    black_adds: ArrayVec<&'a [i16; N], 32>,
    black_rms: ArrayVec<&'a [i16; N], 32>,
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
        let mut updates = Updates::default();
        for color in Color::ALL {
            for piece in Piece::ALL {
                let enabled = &mut self.enabled[color as usize][piece as usize];
                let feats = board.colored_pieces(color, piece);
                let removed = *enabled - feats;
                let added = feats - *enabled;
                *enabled = feats;

                for sq in removed {
                    updates
                        .white_rms
                        .push(&NETWORK.ft.w[feature(Color::White, color, piece, sq)]);
                    updates
                        .black_rms
                        .push(&NETWORK.ft.w[feature(Color::Black, color, piece, sq)]);
                }
                for sq in added {
                    updates
                        .white_adds
                        .push(&NETWORK.ft.w[feature(Color::White, color, piece, sq)]);
                    updates
                        .black_adds
                        .push(&NETWORK.ft.w[feature(Color::Black, color, piece, sq)]);
                }
            }
        }

        #[cfg(feature = "check-inference")]
        let reference = self.clone().infer_scalar(board.side_to_move(), &updates);

        let result = match () {
            #[cfg(target_arch = "x86_64")]
            _ if is_x86_feature_detected!("avx2") => unsafe {
                self.infer_avx2(board.side_to_move(), &updates)
            },
            _ => self.infer_scalar(board.side_to_move(), &updates),
        };

        #[cfg(feature = "check-inference")]
        assert_eq!(result, reference);

        Eval::cp((result / 64).clamp(-29_000, 29_000) as i16)
    }
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
