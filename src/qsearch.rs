use cozy_chess::Board;

use crate::move_picker::MovePicker;
use crate::tt::{Bound, TtSearchEntry};
use crate::{Eval, Search, MAX_PLY};

impl Search<'_> {
    pub(crate) fn qsearch(
        &mut self,
        pos: &Board,
        mut alpha: Eval,
        beta: Eval,
        ply: usize,
    ) -> Option<Eval> {
        self.count_node_and_check_abort(false)?;

        let (tt, tt_eval) = self.shared.tt.load(pos.hash(), ply);
        let tt_mv = tt.map(|tt| tt.mv.into());

        match tt {
            Some(tt) if tt.bound.exact() => return Some(tt.score),
            Some(tt) if tt.bound.lower() && tt.score >= beta => return Some(tt.score),
            Some(tt) if tt.bound.upper() && tt.score <= alpha => return Some(tt.score),
            _ => {}
        }

        let stand_pat = tt_eval.unwrap_or_else(|| {
            let eval = self.eval(pos);
            self.shared.tt.store_eval(pos.hash(), eval);
            eval
        });

        let orig_alpha = alpha;
        let mut best_mv = None;
        let mut best_score = stand_pat;

        if stand_pat > beta || ply >= MAX_PLY {
            return Some(stand_pat);
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut move_picker = MovePicker::new(pos, &self.data, tt_mv, None, true, None, None);

        if !move_picker.has_moves() {
            if pos.checkers().is_empty() {
                return Some(Eval::cp(0));
            } else {
                return Some(Eval::mated(ply));
            }
        }

        while let Some((_, scored_mv)) = move_picker.next(&self.data) {
            if scored_mv.see < 0 {
                continue;
            }

            let mut new_pos = pos.clone();
            new_pos.play_unchecked(scored_mv.mv);

            let score = -self.qsearch(&new_pos, -beta, -alpha, ply + 1)?;

            if score > best_score {
                best_mv = Some(scored_mv.mv);
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
            self.shared.tt.store_search(
                pos.hash(),
                ply,
                TtSearchEntry {
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
