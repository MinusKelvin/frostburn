use alloc::boxed::Box;
use core::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use cozy_chess::{Move, Piece, Square};

use crate::Eval;

pub struct TranspositionTable {
    table: Box<[AtomicU64]>,
    entries: usize,
}

#[derive(Pod, Zeroable, Copy, Clone)]
#[repr(C)]
pub struct TtSearchEntry {
    pub lower_hash_bits: u16,
    pub mv: PackedMove,
    pub score: Eval,
    pub depth: u8,
    pub bound: Bound,
}

#[derive(Pod, Zeroable, Copy, Clone)]
#[repr(C)]
struct TtEvalEntry {
    pub lower_hash_bits: u16,
    pub eval: Eval,
    pub _padding: u32,
}

#[derive(Pod, Zeroable, Copy, Clone)]
#[repr(transparent)]
pub struct PackedMove(u16);

#[derive(Pod, Zeroable, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Bound(u8);

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let entries = mb * 1024 * 1024 / 16;
        TranspositionTable {
            table: bytemuck::zeroed_slice_box(entries * 2),
            entries,
        }
    }

    pub fn load(&self, hash: u64, _ply: usize) -> (Option<TtSearchEntry>, Option<Eval>) {
        let search_data: TtSearchEntry =
            bytemuck::cast(self.slot_search(hash).load(Ordering::Relaxed));
        let eval_data: TtEvalEntry = bytemuck::cast(self.slot_eval(hash).load(Ordering::Relaxed));

        let search_data = (search_data.lower_hash_bits == hash as u16).then_some(search_data);
        let eval_data = (eval_data.lower_hash_bits == hash as u16).then_some(eval_data.eval);

        (search_data, eval_data)
    }

    pub fn store_search(&self, hash: u64, _ply: usize, mut entry: TtSearchEntry) {
        entry.lower_hash_bits = hash as u16;
        entry.score = entry.score.clamp_nonmate();
        self.slot_search(hash)
            .store(bytemuck::cast(entry), Ordering::Relaxed);
    }

    pub fn store_eval(&self, hash: u64, eval: Eval) {
        self.slot_eval(hash).store(
            bytemuck::cast(TtEvalEntry {
                lower_hash_bits: hash as u16,
                eval,
                _padding: 0,
            }),
            Ordering::Relaxed,
        );
    }

    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::*;
            _mm_prefetch::<_MM_HINT_T0>(self.slot_search(hash) as *const _ as *const _);
        }
    }

    pub fn raw(&self) -> &[AtomicU64] {
        &self.table
    }

    fn slot_search(&self, hash: u64) -> &AtomicU64 {
        &self.table[(hash as u128 * self.entries as u128 >> 64) as u64 as usize * 2]
    }

    fn slot_eval(&self, hash: u64) -> &AtomicU64 {
        &self.table[(hash as u128 * self.entries as u128 >> 64) as u64 as usize * 2 + 1]
    }
}

impl From<Move> for PackedMove {
    fn from(value: Move) -> Self {
        PackedMove(
            value.from as u16
                | (value.to as u16) << 6
                | value.promotion.map_or(6, |p| p as u16) << 12,
        )
    }
}

impl From<PackedMove> for Move {
    fn from(value: PackedMove) -> Self {
        Move {
            from: Square::index((value.0 & 0x3F) as usize),
            to: Square::index((value.0 >> 6 & 0x3F) as usize),
            promotion: Piece::try_index((value.0 >> 12 & 0x7) as usize),
        }
    }
}

impl Bound {
    pub const UPPER: Self = Bound(1);
    pub const LOWER: Self = Bound(2);
    pub const EXACT: Self = Bound(3);

    pub const fn exact(self) -> bool {
        self.0 == 3
    }

    pub const fn lower(self) -> bool {
        self.0 == 2
    }

    pub const fn upper(self) -> bool {
        self.0 == 1
    }

    pub const fn lower_or_exact(self) -> bool {
        self.0 & 2 != 0
    }

    // pub const fn upper_or_exact(self) -> bool {
    //     self.0 & 1 != 0
    // }

    pub fn compute(orig_alpha: Eval, beta: Eval, best_score: Eval) -> Self {
        match () {
            _ if best_score <= orig_alpha => Bound::UPPER,
            _ if best_score >= beta => Bound::LOWER,
            _ => Bound::EXACT,
        }
    }
}
