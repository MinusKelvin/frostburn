use std::arch::x86_64::*;

use cozy_chess::Color;

use super::{Accumulator, Updates, NETWORK};

const CHUNKS: usize = 512 / 16;

impl Accumulator {
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn infer_avx2(&mut self, stm: Color, updates: &Updates) -> i32 {
        update(&mut self.white, &updates.white_adds, &updates.white_rms);
        update(&mut self.black, &updates.black_adds, &updates.black_rms);

        let (first, last) = NETWORK.l1.w[0].split_at(512);
        let first = <&[_; 512]>::try_from(first).unwrap();
        let last = <&[_; 512]>::try_from(last).unwrap();

        let (stm, nstm) = match stm {
            Color::White => (&self.white, &self.black),
            Color::Black => (&self.black, &self.white),
        };

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

        NETWORK.l1.bias[0] as i32 + _mm_extract_epi32::<0>(result)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn update(acc: &mut [i16; 512], adds: &[&[i16; 512]], subs: &[&[i16; 512]]) {
    for block in (0..CHUNKS).step_by(16) {
        partial_update(acc, adds, subs, block);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn partial_update(
    acc: &mut [i16; 512],
    adds: &[&[i16; 512]],
    subs: &[&[i16; 512]],
    block: usize,
) {
    let acc: *mut __m256i = acc.as_mut_ptr().cast();
    let mut intermediates = [_mm256_setzero_si256(); 16];
    for i in 0..16 {
        intermediates[i] = _mm256_loadu_si256(acc.add(block + i));
    }
    for add in adds {
        let add: *const __m256i = add.as_ptr().cast();
        for i in 0..16 {
            let v = _mm256_loadu_si256(add.add(block + i));
            intermediates[i] = _mm256_add_epi16(intermediates[i], v);
        }
    }
    for sub in subs {
        let sub: *const __m256i = sub.as_ptr().cast();
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
unsafe fn fused_activate_dot(a: &[i16; 512], w: &[i16; 512]) -> __m256i {
    let a: *const __m256i = a.as_ptr().cast();
    let w: *const __m256i = w.as_ptr().cast();
    let mut result = _mm256_setzero_si256();

    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi16(255);

    for i in 0..CHUNKS {
        let a = _mm256_loadu_si256(a.add(i));
        let a = _mm256_max_epi16(a, zero);
        let a = _mm256_min_epi16(a, one);
        let w = _mm256_loadu_si256(w.add(i));
        let v = _mm256_madd_epi16(a, w);
        result = _mm256_add_epi32(result, v);
    }

    result
}
