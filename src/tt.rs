use core::sync::atomic::{AtomicU64, Ordering};

use alloc::boxed::Box;
use bytemuck::{Pod, Zeroable};
use cozy_chess::{Move, Piece, Square};

use crate::Eval;

pub struct TranspositionTable {
    table: Box<[AtomicU64]>,
}

#[derive(Pod, Zeroable, Copy, Clone)]
#[repr(C)]
pub struct TtEntry {
    pub lower_hash_bits: u16,
    pub mv: PackedMove,
    pub score: Eval,
    pub depth: u8,
    pub bound: Bound,
}

#[derive(Pod, Zeroable, Copy, Clone)]
#[repr(transparent)]
pub struct PackedMove(u16);

#[derive(Pod, Zeroable, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Bound(u8);

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let _: [(); 8] = [(); core::mem::size_of::<TtEntry>()];
        TranspositionTable {
            table: bytemuck::zeroed_slice_box(mb * 1024 * 1024 / core::mem::size_of::<TtEntry>()),
        }
    }

    pub fn load(&self, hash: u64, ply: usize) -> Option<TtEntry> {
        let mut data: TtEntry = bytemuck::cast(self.slot(hash).load(Ordering::Relaxed));
        data.score = data.score.add_time(ply);
        (data.lower_hash_bits == hash as u16).then_some(data)
    }

    pub fn store(&self, hash: u64, ply: usize, mut entry: TtEntry) {
        entry.lower_hash_bits = hash as u16;
        entry.score = entry.score.sub_time(ply);
        self.slot(hash)
            .store(bytemuck::cast(entry), Ordering::Relaxed);
    }

    pub fn raw(&mut self) -> &mut [AtomicU64] {
        &mut self.table
    }

    fn slot(&self, hash: u64) -> &AtomicU64 {
        &self.table[(hash as u128 * self.table.len() as u128 >> 64) as u64 as usize]
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

    // pub const fn lower_or_exact(self) -> bool {
    //     self.0 & 2 != 0
    // }

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
