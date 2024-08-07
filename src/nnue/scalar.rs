use cozy_chess::Color;

use crate::Accumulator;

use super::{Updates, FEATURE_FLIP, NETWORK};

impl Accumulator {
    pub(super) fn infer_scalar(&mut self, stm: Color, updates: &Updates) -> i32 {
        update(&mut self.white, &updates.adds, &updates.rms, 0);
        update(&mut self.black, &updates.adds, &updates.rms, FEATURE_FLIP);

        let mut activated = [0; 1024];
        let (left, right) = activated.split_at_mut(512);
        let left = <&mut [_; 512]>::try_from(left).unwrap();
        let right = <&mut [_; 512]>::try_from(right).unwrap();

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

        for i in 0..activated.len() {
            result += activated[i] as i32 * activated[i] as i32 * NETWORK.l1.w[0][i] as i32;
        }

        result / 256
    }
}

fn update<const N: usize>(acc: &mut [i16; N], adds: &[usize], rms: &[usize], flip: usize) {
    for add in adds {
        let add = &NETWORK.ft.w[add ^ flip];
        for i in 0..N {
            acc[i] += add[i];
        }
    }
    for rm in rms {
        let rm = &NETWORK.ft.w[rm ^ flip];
        for i in 0..N {
            acc[i] -= rm[i];
        }
    }
}

fn crelu<const N: usize>(a: &[i16; N]) -> [i16; N] {
    let mut result = [0; N];
    for i in 0..N {
        result[i] = a[i].clamp(0, 256);
    }
    result
}
