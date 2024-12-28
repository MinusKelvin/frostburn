use super::{Updates, FT_QUANT, HL_SIZE, L1, NETWORK};

pub(super) fn update(acc: &mut [i16; HL_SIZE], updates: &Updates) {
    for &add in &updates.adds {
        let add = &NETWORK.ft.w[add];
        for i in 0..acc.len() {
            acc[i] += add[i];
        }
    }
    for &rm in &updates.rms {
        let rm = &NETWORK.ft.w[rm];
        for i in 0..acc.len() {
            acc[i] -= rm[i];
        }
    }
}

pub(super) fn infer(l1: &L1, stm: &[i16; HL_SIZE], nstm: &[i16; HL_SIZE]) -> i32 {
    let mut activated = [0; HL_SIZE * 2];
    let (left, right) = activated.split_at_mut(HL_SIZE);
    let left = <&mut [_; HL_SIZE]>::try_from(left).unwrap();
    let right = <&mut [_; HL_SIZE]>::try_from(right).unwrap();

    *left = crelu(stm);
    *right = crelu(nstm);

    let mut result = l1.bias[0];

    for i in 0..activated.len() {
        result += (activated[i] as i32 * activated[i] as i32 >> 2) * l1.w[0][i] as i32;
    }

    result
}

fn crelu<const N: usize>(a: &[i16; N]) -> [i16; N] {
    let mut result = [0; N];
    for i in 0..N {
        result[i] = a[i].clamp(0, FT_QUANT);
    }
    result
}
