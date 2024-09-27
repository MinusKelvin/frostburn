use core::arch::x86_64::*;

use super::{Updates, HL_SIZE, NETWORK};

const CHUNKS: usize = HL_SIZE / 16;

const _CHECK_BLOCK_SIZE: () = assert!(
    HL_SIZE % 256 == 0,
    "AVX2 implementation does not support HL sizes which are not a multiple of 256"
);

pub(super) fn available() -> bool {
    cpufeatures::new!(check, "avx2");
    check::get()
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn update(acc: &mut [i16; HL_SIZE], updates: &Updates) {
    for block in (0..CHUNKS).step_by(16) {
        partial_update(acc, &updates.adds, &updates.rms, block);
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn infer(stm: &[i16; HL_SIZE], nstm: &[i16; HL_SIZE]) -> i32 {
    let (first, last) = NETWORK.l1.w[0].split_at(HL_SIZE);
    let first = <&[_; HL_SIZE]>::try_from(first).unwrap();
    let last = <&[_; HL_SIZE]>::try_from(last).unwrap();

    let result = _mm256_add_epi32(
        fused_activate_dot(stm, first),
        fused_activate_dot(nstm, last),
    );

    let mut result = _mm_add_epi32(
        _mm256_extracti128_si256::<0>(result),
        _mm256_extracti128_si256::<1>(result),
    );
    // result = A B C D
    result = _mm_add_epi32(result, _mm_shuffle_epi32::<0b01_00_11_10>(result));
    // result = A+C B+D A+C B+D
    result = _mm_add_epi32(result, _mm_shuffle_epi32::<0b10_11_00_01>(result));
    // result = A+B+C+D A+B+C+D A+B+C+D A+B+C+D

    (NETWORK.l1.bias[0] + _mm_extract_epi32::<0>(result)) / 256 / 64
}

#[target_feature(enable = "avx2")]
unsafe fn partial_update(acc: &mut [i16; HL_SIZE], adds: &[usize], subs: &[usize], block: usize) {
    let acc: *mut __m256i = acc.as_mut_ptr().cast();
    let mut intermediates = [_mm256_setzero_si256(); 16];
    for i in 0..16 {
        intermediates[i] = _mm256_loadu_si256(acc.add(block + i));
    }
    for &add in adds {
        let add: *const __m256i = NETWORK.ft.w[add].as_ptr().cast();
        for i in 0..16 {
            let v = _mm256_loadu_si256(add.add(block + i));
            intermediates[i] = _mm256_add_epi16(intermediates[i], v);
        }
    }
    for &sub in subs {
        let sub: *const __m256i = NETWORK.ft.w[sub].as_ptr().cast();
        for i in 0..16 {
            let v = _mm256_loadu_si256(sub.add(block + i));
            intermediates[i] = _mm256_sub_epi16(intermediates[i], v);
        }
    }
    for i in 0..16 {
        _mm256_storeu_si256(acc.add(block + i), intermediates[i]);
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fused_activate_dot(a: &[i16; HL_SIZE], w: &[i16; HL_SIZE]) -> __m256i {
    let a: *const __m256i = a.as_ptr().cast();
    let w: *const __m256i = w.as_ptr().cast();
    let mut result = _mm256_setzero_si256();

    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi16(256);

    for i in 0..CHUNKS {
        let a = _mm256_loadu_si256(a.add(i));
        let a = _mm256_max_epi16(a, zero);
        let a = _mm256_min_epi16(a, one);

        let w = _mm256_loadu_si256(w.add(i));
        // a: 0..=256, w: -127..=127, therefore a*w fits in i16
        let aw = _mm256_mullo_epi16(a, w);
        let v = _mm256_madd_epi16(a, aw);

        result = _mm256_add_epi32(result, v);
    }

    result
}
