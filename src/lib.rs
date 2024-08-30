// #![no_std]
extern crate alloc;

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::time::Duration;
use std::ops::Range;
use std::sync::atomic::AtomicI16;

use alloc::vec::Vec;
use arrayvec::ArrayVec;
use cozy_chess::{Board, Move, Piece};
use history::{ContinuationHistory, PieceHistory};
use tt::TranspositionTable;

mod eval;
mod history;
mod move_picker;
mod negamax;
mod nnue;
mod params;
mod qsearch;
mod search;
mod tt;

pub use crate::eval::Eval;
pub use crate::nnue::Accumulator;

#[cfg(feature = "tunable")]
pub use crate::params::{Tunable, TUNABLES};

const MAX_PLY: usize = 256;
const MAX_DEPTH: i16 = 120;

pub struct LocalData {
    pv_table: [ArrayVec<Move, MAX_PLY>; MAX_PLY + 1],
    on_first_depth: bool,
    local_nodes: u64,
    local_seldepth: i16,
    accumulator: Accumulator,
    history: PieceHistory,
    counter_hist: ContinuationHistory,
    followup_hist: ContinuationHistory,
    prev_moves: [Option<(Move, Piece)>; MAX_PLY],
    prev_evals: [Eval; MAX_PLY],
}

pub struct SharedData {
    abort: AtomicBool,
    nodes: AtomicU64,
    selective_depth: AtomicI16,
    tt: TranspositionTable,
    log_table: [f32; 32],
    pub seed: u64,
}

pub struct Search<'a> {
    pub root: &'a Board,
    pub history: Vec<u64>,
    pub clock: &'a dyn Fn() -> Duration,
    pub info: &'a mut dyn FnMut(SearchInfo),
    pub data: &'a mut LocalData,
    pub shared: &'a SharedData,
    pub limits: Limits,
}

#[derive(Copy, Clone)]
pub struct Limits {
    pub move_time: Option<Duration>,
    pub clock: Option<Duration>,
    pub increment: Duration,
    pub depth: Option<i16>,
    pub nodes: Option<u64>,
    pub min_nodes: Option<u64>,
    pub quantize_eval: i16,
}

impl Limits {
    pub fn unbounded(&mut self) {
        self.move_time = None;
        self.clock = None;
        self.depth = None;
        self.nodes = None;
        self.min_nodes = None;
        self.increment = Duration::ZERO;
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            move_time: None,
            clock: None,
            increment: Duration::ZERO,
            depth: None,
            nodes: None,
            min_nodes: None,
            quantize_eval: 1,
        }
    }
}

pub struct SearchInfo<'a> {
    pub depth: i16,
    pub selective_depth: i16,
    pub score: Eval,
    pub nodes: u64,
    pub time: Duration,
    pub pv: &'a [Move],
    pub finished: bool,
}

impl Search<'_> {
    fn count_node_and_check_abort(&mut self, force_node_accumulate: bool) -> Option<()> {
        self.data.local_nodes += 1;

        if force_node_accumulate
            || self.limits.nodes.is_some()
            || self.shared.abort.load(Ordering::Relaxed)
            || self.data.local_nodes == 1024
        {
            let nodes = self.data.local_nodes
                + self
                    .shared
                    .nodes
                    .fetch_add(self.data.local_nodes, Ordering::Relaxed);
            self.data.local_nodes = 0;

            if self.limits.nodes.is_some_and(|n| nodes >= n) || self.check_time() {
                self.shared.abort.store(true, Ordering::Relaxed);
            }
        }

        if !self.data.on_first_depth && self.shared.abort.load(Ordering::Relaxed) {
            return None;
        }

        Some(())
    }

    fn check_time(&self) -> bool {
        let time = (self.clock)();
        if self.limits.move_time.is_some_and(|target| time > target) {
            return true;
        }
        false
    }

    fn eval(&mut self, board: &Board) -> Eval {
        let mut eval = self.data.accumulator.infer(board);
        if self.limits.quantize_eval != 1 {
            let offset = self.limits.quantize_eval / 2 * eval.signum();
            eval = (eval + offset) / self.limits.quantize_eval * self.limits.quantize_eval;
        }
        Eval::cp(eval)
    }
}

impl LocalData {
    pub fn new() -> Self {
        Self {
            pv_table: [(); MAX_PLY + 1].map(|_| ArrayVec::new()),
            on_first_depth: false,
            local_nodes: 0,
            local_seldepth: 0,
            accumulator: Accumulator::new(),
            history: PieceHistory::new(),
            counter_hist: ContinuationHistory::new(),
            followup_hist: ContinuationHistory::new(),
            prev_moves: [None; MAX_PLY],
            prev_evals: [Eval::cp(0); MAX_PLY],
        }
    }
}

impl SharedData {
    pub fn new(tt_mb: usize) -> Self {
        let mut log_table = [0.0; 32];
        for i in 1..32 {
            log_table[i] = (i as f32).ln();
        }
        SharedData {
            abort: AtomicBool::new(false),
            nodes: AtomicU64::new(0),
            selective_depth: AtomicI16::new(0),
            tt: TranspositionTable::new(tt_mb),
            seed: 0x6CA648710DB5F3AE,
            log_table,
        }
    }

    pub fn prepare_for_search(&mut self) {
        *self.abort.get_mut() = false;
        *self.nodes.get_mut() = 0;
        *self.selective_depth.get_mut() = 0;
    }

    pub fn abort(&self) {
        self.abort.store(true, Ordering::SeqCst);
    }

    pub fn get_clear_tt_blocks(&self, count: usize) -> Vec<ClearTtBlock> {
        let size = self.tt.raw().len();
        let block_size = size / count;
        let mut big = size - block_size * count;
        let mut i = 0;
        let mut blocks = vec![];
        for _ in 0..count {
            let mut upper = i + block_size;
            if big > 0 {
                big -= 1;
                upper += 1;
            }
            blocks.push(ClearTtBlock { range: i..upper });
            i = upper;
        }
        blocks
    }

    pub fn clear_tt_block(&self, block: ClearTtBlock) {
        for slot in &self.tt.raw()[block.range] {
            slot.store(0, Ordering::Relaxed);
        }
    }

    fn log(&self, i: usize) -> f32 {
        self.log_table[i.min(self.log_table.len() - 1)]
    }
}

pub struct ClearTtBlock {
    range: Range<usize>,
}
