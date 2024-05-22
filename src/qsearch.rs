use alloc::vec;
use cozy_chess::Board;

use crate::tt::{Bound, TtEntry};
use crate::{Search, MAX_PLY};

impl Search<'_> {
    pub(crate) fn qsearch(
        &mut self,
        pos: &Board,
        mut alpha: i16,
        beta: i16,
        ply: usize,
    ) -> Option<i16> {
        self.count_node_and_check_abort(false)?;

        let tt = self.shared.tt.load(pos.hash());
        let tt_mv = tt.map(|tt| tt.mv.into());

        match tt {
            Some(tt) if tt.bound.exact() => return Some(tt.score),
            Some(tt) if tt.bound.lower() && tt.score >= beta => return Some(tt.score),
            Some(tt) if tt.bound.upper() && tt.score <= alpha => return Some(tt.score),
            _ => {}
        }

        let stand_pat = self.data.accumulator.infer(pos);

        let mut best_mv = None;
        let mut best_score = stand_pat;
        let orig_alpha = alpha;

        if stand_pat > beta || ply >= MAX_PLY {
            return Some(stand_pat);
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut moves = vec![];
        let mut has_moves = false;
        pos.generate_moves(|mvs| {
            has_moves = true;
            for mv in mvs {
                if !pos.colors(!pos.side_to_move()).has(mv.to) {
                    continue;
                }
                let score = if tt_mv.is_some_and(|tt_mv| mv == tt_mv) {
                    1_000_000
                } else {
                    100_000 + pos.piece_on(mv.to).unwrap() as i32
                };
                moves.push((mv, score));
            }
            false
        });

        if !has_moves {
            if pos.checkers().is_empty() {
                return Some(0);
            } else {
                return Some(ply as i16 - 30_000);
            }
        }

        moves.sort_unstable_by_key(|&(_, score)| core::cmp::Reverse(score));

        for (mv, _) in moves {
            let mut new_pos = pos.clone();
            new_pos.play_unchecked(mv);

            let score = -self.qsearch(&new_pos, -beta, -alpha, ply + 1)?;

            if score > best_score {
                best_mv = Some(mv);
                best_score = score;
            }

            if score > alpha {
                alpha = score;
            }

            if score > beta {
                break;
            }
        }

        if let Some(best_mv) = best_mv {
            self.shared.tt.store(
                pos.hash(),
                TtEntry {
                    lower_hash_bits: 0,
                    mv: best_mv.into(),
                    score: best_score,
                    depth: 0,
                    bound: Bound::compute(orig_alpha, beta, best_score),
                },
            );
        }

        Some(best_score)
    }
}
