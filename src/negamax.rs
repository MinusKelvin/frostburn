use alloc::vec;
use cozy_chess::Board;

use crate::{Search, MAX_PLY};

impl Search<'_> {
    pub(crate) fn negamax(
        &mut self,
        pos: &Board,
        mut alpha: i16,
        beta: i16,
        depth: i16,
        ply: usize,
    ) -> Option<i16> {
        if depth <= 0 || ply >= MAX_PLY {
            return self.qsearch(pos, alpha, beta, ply);
        }

        self.count_node_and_check_abort(false)?;

        let mut best_mv = None;
        let mut best_score = -30_000;

        let mut moves = vec![];
        pos.generate_moves(|mvs| {
            moves.extend(mvs);
            false
        });

        moves.sort_unstable_by_key(|mv| core::cmp::Reverse(pos.piece_on(mv.to)));

        self.history.push(pos.hash());

        for mv in moves {
            let mut new_pos = pos.clone();
            new_pos.play_unchecked(mv);
            self.data.pv_table[ply + 1].clear();

            let score;
            if ply != 0 && self.history.contains(&new_pos.hash()) {
                score = 0;
            } else {
                score = -self.negamax(&new_pos, -beta, -alpha, depth - 1, ply + 1)?;
            }

            if score > best_score {
                best_mv = Some(mv);
                best_score = score;

                let [pv, cont] = self.data.pv_table[ply..].first_chunk_mut().unwrap();
                pv.clear();
                pv.push(mv);
                pv.try_extend_from_slice(&cont).unwrap();
            }

            if score > alpha {
                alpha = score;
            }

            if score > beta {
                break;
            }
        }

        self.history.pop();

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
