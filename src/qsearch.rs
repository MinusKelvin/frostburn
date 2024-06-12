use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves, BitBoard,
    Board, Color, Move, Piece, Square,
};

use crate::move_picker::MovePicker;
use crate::tt::{Bound, TtEntry};
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

        let tt = self.shared.tt.load(pos.hash(), ply);
        let tt_mv = tt.map(|tt| tt.mv.into());

        match tt {
            Some(tt) if tt.bound.exact() => return Some(tt.score),
            Some(tt) if tt.bound.lower() && tt.score >= beta => return Some(tt.score),
            Some(tt) if tt.bound.upper() && tt.score <= alpha => return Some(tt.score),
            _ => {}
        }

        let stand_pat = self.data.accumulator.infer(pos);

        let orig_alpha = alpha;
        let mut best_mv = None;
        let mut best_score = stand_pat;

        if stand_pat > beta || ply >= MAX_PLY {
            return Some(stand_pat);
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut move_picker = MovePicker::new(pos, &self.data, tt_mv, true, None, None);

        if !move_picker.has_moves() {
            if pos.checkers().is_empty() {
                return Some(Eval::cp(0));
            } else {
                return Some(Eval::mated(ply));
            }
        }

        while let Some((_, mv, _)) = move_picker.next(&self.data) {
            if see(pos, mv) < 0 {
                continue;
            }

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
                ply,
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

fn see(pos: &Board, mv: Move) -> i32 {
    const VALUES: [i32; 6] = [100, 300, 325, 500, 900, 0];

    fn see_impl(pos: &Board, occupied: BitBoard, sq: Square, stm: Color, piece: Piece) -> i32 {
        let mut attacker = None;

        if attacker.is_none() {
            let pawns =
                occupied & get_pawn_attacks(sq, !stm) & pos.colored_pieces(stm, Piece::Pawn);
            attacker = pawns.next_square();
        }

        if attacker.is_none() {
            let knights = occupied & get_knight_moves(sq) & pos.colored_pieces(stm, Piece::Knight);
            attacker = knights.next_square();
        }

        if attacker.is_none() {
            let bishops =
                occupied & get_bishop_moves(sq, occupied) & pos.colored_pieces(stm, Piece::Bishop);
            attacker = bishops.next_square();
        }

        if attacker.is_none() {
            let rooks =
                occupied & get_rook_moves(sq, occupied) & pos.colored_pieces(stm, Piece::Rook);
            attacker = rooks.next_square();
        }
        if attacker.is_none() {
            let queens = occupied
                & (get_rook_moves(sq, occupied) | get_bishop_moves(sq, occupied))
                & pos.colored_pieces(stm, Piece::Queen);
            attacker = queens.next_square();
        }

        if attacker.is_none() {
            let kings = occupied & get_king_moves(sq) & pos.colored_pieces(stm, Piece::King);
            attacker = kings.next_square();
        }

        if let Some(atk) = attacker {
            return 0.max(
                VALUES[piece as usize]
                    - see_impl(
                        pos,
                        occupied - atk.bitboard(),
                        sq,
                        !stm,
                        pos.piece_on(atk).unwrap(),
                    ),
            );
        }

        0
    }

    let captured = pos
        .piece_on(mv.to)
        .map_or(0, |piece| VALUES[piece as usize]);
    let occupied = pos.occupied() - mv.from.bitboard();

    captured
        - see_impl(
            pos,
            occupied,
            mv.to,
            !pos.side_to_move(),
            pos.piece_on(mv.from).unwrap(),
        )
}
