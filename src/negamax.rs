use alloc::vec;
use cozy_chess::Board;

use crate::{Search, MAX_PLY};

impl Search<'_> {
    pub(crate) fn negamax(&mut self, pos: &Board, depth: i16, ply: usize) -> Option<i16> {
        self.data.pv_table[ply].clear();

        if depth <= 0 || ply >= MAX_PLY {
            return Some((pos.hash() % 8192) as i16 - 4096);
        }

        self.count_node_and_check_abort(false)?;

        let mut best_mv = None;
        let mut best_score = -30_000;

        let mut moves = vec![];
        pos.generate_moves(|mvs| {
            moves.extend(mvs);
            false
        });

        for mv in moves {
            let mut new_pos = pos.clone();
            new_pos.play_unchecked(mv);

            let score = -self.negamax(&new_pos, depth - 1, ply + 1)?;

            if score > best_score {
                best_mv = Some(mv);
                best_score = score;
                let [pv, cont] = self.data.pv_table[ply..].first_chunk_mut().unwrap();
                pv.clear();
                pv.push(mv);
                pv.try_extend_from_slice(&cont).unwrap();
            }
        }

        if best_mv.is_none() {
            if pos.checkers().is_empty() {
                return Some(0);
            } else {
                return Some(ply as i16 - 30_000);
            }
        }

        Some(best_score)
    }
}
