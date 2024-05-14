use cozy_chess::{Board, Color, Piece, Square};

#[derive(Clone)]
pub struct Accumulator {
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
    pub fn new(pos: &Board) -> Self {
        let mut this = Accumulator {
            white: NETWORK.ft.bias,
            black: NETWORK.ft.bias,
        };
        for sq in pos.occupied() {
            let color = pos.color_on(sq).unwrap();
            let piece = pos.piece_on(sq).unwrap();
            let w_feat = feature(Color::White, color, piece, sq);
            let b_feat = feature(Color::Black, color, piece, sq);
            this.white = vadd(&this.white, &NETWORK.ft.w[w_feat]);
            this.black = vadd(&this.black, &NETWORK.ft.w[b_feat]);
        }
        this
    }

    pub fn infer(&self, stm: Color) -> i16 {
        let mut activated = [0; 512];
        let (left, right) = activated.split_at_mut(256);
        let left = <&mut [_; 256]>::try_from(left).unwrap();
        let right = <&mut [_; 256]>::try_from(right).unwrap();
        match stm {
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

        (result / 128).clamp(-29_000, 29_000) as i16
    }
}

fn vadd<const N: usize>(a: &[i16; N], b: &[i16; N]) -> [i16; N] {
    let mut result = [0; N];
    for i in 0..N {
        result[i] = a[i] + b[i];
    }
    result
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
