use core::arch::x86_64::*;

use super::{Updates, HL_SIZE, NETWORK};

const NEURONS_PER_VECTOR: usize = 512 / 16;
const VECTORS_PER_BLOCK: usize = 16;

const HL_VECTORS: usize = HL_SIZE / NEURONS_PER_VECTOR;

const _: () = {
    const NEURONS_PER_BLOCK: usize = NEURONS_PER_VECTOR * VECTORS_PER_BLOCK;
    assert!(
        HL_SIZE % NEURONS_PER_BLOCK == 0,
        "AVX2 implementation does not support HL sizes which are not a multiple of 1024"
    )
};

pub(super) fn available() -> bool {
    cpufeatures::new!(check, "avx512f", "avx512bw");
    check::get()
}

#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn update(acc: &mut [i16; HL_SIZE], updates: &Updates) {
    for block in (0..HL_VECTORS).step_by(VECTORS_PER_BLOCK) {
        partial_update(acc, &updates.adds, &updates.rms, block);
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn infer(stm: &[i16; HL_SIZE], nstm: &[i16; HL_SIZE]) -> i32 {
    let (first, last) = NETWORK.l1.w[0].split_at(HL_SIZE);
    let first = <&[_; HL_SIZE]>::try_from(first).unwrap();
    let last = <&[_; HL_SIZE]>::try_from(last).unwrap();

    let result = _mm512_add_epi32(
        fused_activate_dot(stm, first),
        fused_activate_dot(nstm, last),
    );

    let result = _mm512_reduce_add_epi32(result);

    NETWORK.l1.bias[0] + result
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn partial_update(acc: &mut [i16; HL_SIZE], adds: &[usize], subs: &[usize], block: usize) {
    let acc: *mut __m512i = acc.as_mut_ptr().cast();
    let mut intermediates = [_mm512_setzero_si512(); VECTORS_PER_BLOCK];
    for i in 0..VECTORS_PER_BLOCK {
        intermediates[i] = _mm512_loadu_si512(acc.add(block + i).cast());
    }
    for &add in adds {
        let add: *const __m512i = NETWORK.ft.w[add].as_ptr().cast();
        for i in 0..VECTORS_PER_BLOCK {
            let v = _mm512_loadu_si512(add.add(block + i).cast());
            intermediates[i] = _mm512_add_epi16(intermediates[i], v);
        }
    }
    for &sub in subs {
        let sub: *const __m512i = NETWORK.ft.w[sub].as_ptr().cast();
        for i in 0..VECTORS_PER_BLOCK {
            let v = _mm512_loadu_si512(sub.add(block + i).cast());
            intermediates[i] = _mm512_sub_epi16(intermediates[i], v);
        }
    }
    for i in 0..VECTORS_PER_BLOCK {
        _mm512_storeu_si512(acc.add(block + i).cast(), intermediates[i]);
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn fused_activate_dot(a: &[i16; HL_SIZE], w: &[i16; HL_SIZE]) -> __m512i {
    let a: *const __m512i = a.as_ptr().cast();
    let w: *const __m512i = w.as_ptr().cast();
    let mut result = _mm512_setzero_si512();

    let zero = _mm512_setzero_si512();
    let one = _mm512_set1_epi16(256);

    for i in 0..HL_VECTORS {
        let a = _mm512_loadu_si512(a.add(i).cast());
        let a = _mm512_max_epi16(a, zero);
        let a = _mm512_min_epi16(a, one);

        let w = _mm512_loadu_si512(w.add(i).cast());
        // a: 0..=256, w: -127..=127, therefore a*w fits in i16
        let aw = _mm512_mullo_epi16(a, w);
        let v = _mm512_madd_epi16(a, aw);

        result = _mm512_add_epi32(result, v);
    }

    result
}
