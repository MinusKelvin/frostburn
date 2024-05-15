use alloc::vec;
use cozy_chess::Board;

use crate::nnue::Accumulator;
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

        let stand_pat = Accumulator::new(pos).infer(pos.side_to_move());

        let mut best_mv = None;
        let mut best_score = stand_pat;

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
            moves.extend(
                mvs.into_iter()
                    .filter(|mv| pos.colors(!pos.side_to_move()).has(mv.to)),
            );
            false
        });

        if !has_moves {
            if pos.checkers().is_empty() {
                return Some(0);
            } else {
                return Some(ply as i16 - 30_000);
            }
        }

        moves.sort_unstable_by_key(|mv| core::cmp::Reverse(pos.piece_on(mv.to)));

        for mv in moves {
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

        Some(best_score)
    }
}
