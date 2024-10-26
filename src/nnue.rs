use alloc::vec::Vec;
use alloc::vec;

use arrayvec::ArrayVec;
use cozy_chess::{BitBoard, Board, Color, File, Piece, Square};

#[cfg(target_arch = "x86_64")]
mod avx2;

mod scalar;

const HL_SIZE: usize = 512;

const BLACK_FLIP: usize = 0b1_111_000;
const MIRROR_FLIP: usize = 0b0_000_111;

// Note: This type has the safety invariant that the contained `Backend` is safe to use.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NnueBackend(Backend);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Backend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

pub struct Nnue {
    white_left: Accumulator,
    white_right: Accumulator,
    black_left: Accumulator,
    black_right: Accumulator,
}

#[derive(Clone)]
struct Accumulator {
    flip: usize,
    enabled: [[BitBoard; 6]; 2],
    vector: [i16; HL_SIZE],
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
    ft: FeatureTransformer<768, HL_SIZE>,
    l1: Linear<{ 2 * HL_SIZE }, 1>,
}

#[derive(Default)]
struct Updates {
    adds: ArrayVec<usize, 32>,
    rms: ArrayVec<usize, 32>,
}

impl NnueBackend {
    pub fn available() -> Vec<NnueBackend> {
        let mut backends = vec![NnueBackend(Backend::Scalar)];

        #[cfg(target_arch = "x86_64")]
        if avx2::available() {
            backends.push(NnueBackend(Backend::Avx2));
        }

        backends
    }

    pub fn name(self) -> &'static str {
        match self.0 {
            Backend::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => "avx2",
        }
    }
}

impl Default for NnueBackend {
    fn default() -> Self {
        NnueBackend::available().into_iter().max().unwrap()
    }
}

impl Nnue {
    pub fn new() -> Self {
        Nnue {
            white_left: Accumulator::new(0),
            white_right: Accumulator::new(MIRROR_FLIP),
            black_left: Accumulator::new(BLACK_FLIP),
            black_right: Accumulator::new(BLACK_FLIP | MIRROR_FLIP),
        }
    }

    pub fn infer(&mut self, board: &Board, backend: NnueBackend) -> i32 {
        let white_acc = match board.king(Color::White).file() < File::E {
            true => &mut self.white_left,
            false => &mut self.white_right,
        };
        let black_acc = match board.king(Color::Black).file() < File::E {
            true => &mut self.black_left,
            false => &mut self.black_right,
        };

        white_acc.update(board, backend.0);
        black_acc.update(board, backend.0);

        let (stm_acc, nstm_acc) = match board.side_to_move() {
            Color::White => (white_acc, black_acc),
            Color::Black => (black_acc, white_acc),
        };

        let result = match backend.0 {
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => unsafe { avx2::infer(&stm_acc.vector, &nstm_acc.vector) },
            Backend::Scalar => scalar::infer(&stm_acc.vector, &nstm_acc.vector),
        };

        #[cfg(feature = "check-inference")]
        assert_eq!(scalar::infer(&stm_acc.vector, &nstm_acc.vector), result);

        result
    }
}

impl Accumulator {
    pub fn new(flip: usize) -> Self {
        Accumulator {
            flip,
            enabled: [[BitBoard::EMPTY; 6]; 2],
            vector: NETWORK.ft.bias,
        }
    }

    fn update(&mut self, board: &Board, backend: Backend) {
        let mut updates = Updates::default();
        for color in 0..Color::NUM {
            let color = Color::index(color);
            for piece in 0..Piece::NUM {
                let piece = Piece::index(piece);
                let enabled = &mut self.enabled[color as usize][piece as usize];
                let feats = board.colored_pieces(color, piece);
                if *enabled == feats {
                    continue;
                }
                let removed = *enabled - feats;
                let added = feats - *enabled;
                *enabled = feats;

                for sq in removed {
                    updates.rms.push(feature(color, piece, sq) ^ self.flip);
                }
                for sq in added {
                    updates.adds.push(feature(color, piece, sq) ^ self.flip);
                }
            }
        }

        #[cfg(feature = "check-inference")]
        let reference = {
            let mut reference = self.clone();
            scalar::update(&mut reference.vector, &updates);
            reference
        };

        match backend {
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => unsafe { avx2::update(&mut self.vector, &updates) },
            Backend::Scalar => scalar::update(&mut self.vector, &updates),
        };

        #[cfg(feature = "check-inference")]
        assert_eq!(self.vector, reference.vector);
    }
}

fn feature(color: Color, piece: Piece, sq: Square) -> usize {
    let i = 0;
    let i = i * Piece::NUM + piece as usize;
    let i = i * Color::NUM + color as usize;
    let i = i * Square::NUM + sq as usize;
    i
}

#[cfg(target_endian = "little")]
static NETWORK: Network = unsafe { core::mem::transmute(*include_bytes!(env!("NNUE_PATH"))) };

#[cfg(not(target_endian = "little"))]
compile_error!("Non-little endian targets are not currently supported.");
