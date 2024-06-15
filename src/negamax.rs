use cozy_chess::{Board, Move, Square};

use crate::move_picker::MovePicker;
use crate::params::*;
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

        let depth = match tt.is_some() {
            false if depth > 3 => depth - 1,
            _ => depth,
        };

        let static_eval = self.data.accumulator.infer(pos);

        let eval = tt.map_or(static_eval, |tt| tt.score);

        if !PV
            && pos.checkers().is_empty()
            && depth <= rfp_max_depth()
            && eval >= beta + rfp_margin() * depth
        {
            return Some(eval);
        }

        if !PV && pos.checkers().is_empty() && eval >= beta && depth >= nmp_min_depth() {
            let new_pos = pos.null_move().unwrap();
            let r = (eval - beta + depth * nmp_depth() + nmp_constant()) / nmp_divisor();
            self.data.prev_moves[ply] = None;
            let score = self.search_opp::<false>(&new_pos, beta - 1, beta, depth - r, ply + 1)?;
            if score >= beta {
                return Some(score);
            }
        }

        let counter_prior = (ply > 0).then(|| self.data.prev_moves[ply - 1]).flatten();
        let followup_prior = (ply > 1).then(|| self.data.prev_moves[ply - 2]).flatten();

        let orig_alpha = alpha;
        let mut best_mv = None;
        let mut best_score = Eval::mated(0);
        let mut move_picker = MovePicker::new(
            pos,
            &self.data,
            tt_mv,
            false,
            self.data.counter_hist.get(counter_prior),
            self.data.followup_hist.get(followup_prior),
        );

        self.history.push(pos.hash());

        while let Some((i, scored_mv)) = move_picker.next(&self.data) {
            let piece = pos.piece_on(scored_mv.mv.from).unwrap();

            let quiet = !pos.colors(!pos.side_to_move()).has(scored_mv.mv.to);

            if !PV && quiet && i > depth as usize * depth as usize + 4 {
                continue;
            }

            let mut new_pos = pos.clone();
            new_pos.play_unchecked(scored_mv.mv);
            self.data.pv_table[ply + 1].clear();
            self.data.prev_moves[ply] = Some((scored_mv.mv, piece));

            let mut score;
            if ply != 0 && self.history.contains(&new_pos.hash()) {
                score = Eval::cp(0);
            } else if PV && i == 0 {
                score = self.search_opp::<true>(&new_pos, alpha, beta, depth - 1, ply + 1)?;
            } else {
                let log_log = self.shared.log(i) * self.shared.log(depth as usize);
                let mut r = match quiet {
                    true => log_log / 1.5 + 0.25,
                    false => log_log * 0.5 - 0.5,
                } as i16;

                r -= (scored_mv.history / 4096).clamp(-4, 4) as i16;
                r -= PV as i16;

                if r < 0 {
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
                best_mv = Some(scored_mv.mv);
                best_score = score;

                if PV {
                    let [pv, cont] = self.data.pv_table[ply..].first_chunk_mut().unwrap();
                    pv.clear();
                    pv.push(scored_mv.mv);
                    pv.try_extend_from_slice(&cont).unwrap();
                }
            }

            if score > alpha {
                alpha = score;
            }

            if score > beta {
                if quiet {
                    let mut counter_hist = self.data.counter_hist.get_mut(counter_prior);
                    let mut followup_hist = self.data.followup_hist.get_mut(followup_prior);

                    self.data.history.update(pos, scored_mv.mv, 64 * depth);
                    if let Some(counter_hist) = counter_hist.as_deref_mut() {
                        counter_hist.update(pos, scored_mv.mv, 64 * depth);
                    }
                    if let Some(followup_hist) = followup_hist.as_deref_mut() {
                        followup_hist.update(pos, scored_mv.mv, 64 * depth);
                    }

                    for failure in move_picker.failed() {
                        let failure = failure.mv;
                        if !pos.colors(!pos.side_to_move()).has(failure.to) {
                            self.data.history.update(pos, failure, -64 * depth);
                            if let Some(counter_hist) = counter_hist.as_deref_mut() {
                                counter_hist.update(pos, failure, -64 * depth);
                            }
                            if let Some(followup_hist) = followup_hist.as_deref_mut() {
                                followup_hist.update(pos, failure, -64 * depth);
                            }
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

        let bound = Bound::compute(orig_alpha, beta, best_score);
        self.shared.tt.store(
            pos.hash(),
            ply,
            TtEntry {
                lower_hash_bits: 0,
                mv: match bound {
                    Bound::UPPER => tt_mv.unwrap_or(Move {
                        from: Square::A1,
                        to: Square::A1,
                        promotion: None,
                    }),
                    _ => best_mv.into(),
                }
                .into(),
                score: best_score,
                depth: depth as u8,
                bound,
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
