// #![no_std]
extern crate alloc;

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::time::Duration;

use alloc::vec::Vec;
use arrayvec::ArrayVec;
use cozy_chess::{Board, Move};
use history::PieceHistory;
use tt::TranspositionTable;

mod negamax;
mod nnue;
mod qsearch;
mod search;
mod tt;
mod history;
mod eval;

pub use crate::nnue::Accumulator;
pub use crate::eval::Eval;

const MAX_PLY: usize = 256;
const MAX_DEPTH: i16 = 120;

pub struct LocalData {
    pv_table: [ArrayVec<Move, MAX_PLY>; MAX_PLY + 1],
    on_first_depth: bool,
    local_nodes: u64,
    accumulator: Accumulator,
    history: PieceHistory,
}

pub struct SharedData {
    abort: AtomicBool,
    nodes: AtomicU64,
    tt: TranspositionTable,
    log_table: [f32; 32],
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

#[derive(Default, Copy, Clone)]
pub struct Limits {
    pub move_time: Option<Duration>,
    pub depth: Option<i16>,
    pub nodes: Option<u64>,
    pub min_nodes: Option<u64>,
}

pub struct SearchInfo<'a> {
    pub depth: i16,
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
}

impl LocalData {
    pub fn new() -> Self {
        Self {
            pv_table: [(); MAX_PLY + 1].map(|_| ArrayVec::new()),
            on_first_depth: false,
            local_nodes: 0,
            accumulator: Accumulator::new(),
            history: PieceHistory::new(),
        }
    }
}

impl SharedData {
    pub fn new(tt_mb: usize) -> Self {
        let mut log_table = [0.0; 32];
        for i in 0..32 {
            log_table[i] = (i as f32).ln();
        }
        SharedData {
            abort: AtomicBool::new(false),
            nodes: AtomicU64::new(0),
            tt: TranspositionTable::new(tt_mb),
            log_table,
        }
    }

    pub fn prepare_for_search(&mut self) {
        *self.abort.get_mut() = false;
        *self.nodes.get_mut() = 0;
    }

    pub fn abort(&self) {
        self.abort.store(true, Ordering::SeqCst);
    }

    fn log(&self, i: usize) -> f32 {
        self.log_table[i.min(self.log_table.len() - 1)]
    }
}
