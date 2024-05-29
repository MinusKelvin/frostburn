use alloc::vec;
use cozy_chess::Board;

use crate::tt::{Bound, TtEntry};
use crate::{Eval, Search, MAX_PLY};

impl Search<'_> {
    pub(crate) fn negamax<const PV: bool>(
        &mut self,
        pos: &Board,
        mut alpha: Eval,
        beta: Eval,
        depth: i16,
        ply: usize,
    ) -> Option<Eval> {
        if depth <= 0 || ply >= MAX_PLY {
            return self.qsearch(pos, alpha, beta, ply);
        }

        self.count_node_and_check_abort(false)?;

        let tt = self.shared.tt.load(pos.hash(), ply);
        let tt_mv = tt.map(|tt| tt.mv.into());

        match tt {
            _ if PV => {}
            Some(tt) if depth > tt.depth as i16 => {}
            Some(tt) if tt.bound.exact() => return Some(tt.score),
            Some(tt) if tt.bound.lower() && tt.score >= beta => return Some(tt.score),
            Some(tt) if tt.bound.upper() && tt.score <= alpha => return Some(tt.score),
            _ => {}
        }

        let static_eval = self.data.accumulator.infer(pos);

        if !PV && pos.checkers().is_empty() && depth < 5 && static_eval >= beta + 50 * depth {
            return Some(static_eval);
        }

        if !PV && pos.checkers().is_empty() {
            let new_pos = pos.null_move().unwrap();
            let score = self.search_opp::<false>(&new_pos, beta - 1, beta, depth - 4, ply + 1)?;
            if score >= beta {
                return Some(score);
            }
        }

        let mut best_mv = None;
        let mut best_score = Eval::mated(0);
        let orig_alpha = alpha;

        let mut moves = vec![];
        pos.generate_moves(|mvs| {
            for mv in mvs {
                let score = if tt_mv.is_some_and(|tt_mv| mv == tt_mv) {
                    1_000_000
                } else if pos.colors(!pos.side_to_move()).has(mv.to) {
                    100_000 + pos.piece_on(mv.to).unwrap() as i32
                } else {
                    self.data.history.get(pos, mv) as i32
                };
                moves.push((mv, score));
            }
            false
        });

        moves.sort_unstable_by_key(|&(_, score)| core::cmp::Reverse(score));

        self.history.push(pos.hash());

        for (i, &(mv, mv_score)) in moves.iter().enumerate() {
            if !PV && mv_score < 100_000 && i > depth as usize * depth as usize + 4 {
                continue;
            }

            let mut new_pos = pos.clone();
            new_pos.play_unchecked(mv);
            self.data.pv_table[ply + 1].clear();

            let mut score;
            if ply != 0 && self.history.contains(&new_pos.hash()) {
                score = Eval::cp(0);
            } else if PV && i == 0 {
                score = self.search_opp::<true>(&new_pos, alpha, beta, depth - 1, ply + 1)?;
            } else {
                let mut r = i as i16 / 4;

                if r < 0 || pos.colors(!pos.side_to_move()).has(mv.to) {
                    r = 0;
                }

                score =
                    self.search_opp::<false>(&new_pos, alpha, alpha + 1, depth - r - 1, ply + 1)?;

                if r > 0 && score > alpha {
                    score =
                        self.search_opp::<false>(&new_pos, alpha, alpha + 1, depth - 1, ply + 1)?;
                }

                if PV && score > alpha {
                    score = self.search_opp::<true>(&new_pos, alpha, beta, depth - 1, ply + 1)?;
                }
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
                if !pos.colors(!pos.side_to_move()).has(mv.to) {
                    self.data.history.update(pos, mv, 64 * depth);
                    for &(failure, _) in &moves[..i] {
                        if !pos.colors(!pos.side_to_move()).has(failure.to) {
                            self.data.history.update(pos, failure, -64 * depth);
                        }
                    }
                }
                break;
            }
        }

        self.history.pop();

        let Some(best_mv) = best_mv else {
            if pos.checkers().is_empty() {
                return Some(Eval::cp(0));
            } else {
                return Some(Eval::mated(ply));
            }
        };

        self.shared.tt.store(
            pos.hash(),
            ply,
            TtEntry {
                lower_hash_bits: 0,
                mv: best_mv.into(),
                score: best_score,
                depth: depth as u8,
                bound: Bound::compute(orig_alpha, beta, best_score),
            },
        );

        Some(best_score)
    }

    fn search_opp<const PV: bool>(
        &mut self,
        pos: &Board,
        alpha: Eval,
        beta: Eval,
        depth: i16,
        ply: usize,
    ) -> Option<Eval> {
        self.negamax::<PV>(pos, -beta, -alpha, depth, ply)
            .map(|e| -e)
    }
}
