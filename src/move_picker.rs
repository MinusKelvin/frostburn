use arrayvec::ArrayVec;
use cozy_chess::{Board, Move, Piece};

use crate::history::PieceHistory;
use crate::LocalData;

pub struct MovePicker<'a> {
    _skip_quiets: bool,
    has_moves: bool,
    _board: &'a Board,
    moves: Vec<(Move, i32)>,
    next_idx: usize,
}

impl<'a> MovePicker<'a> {
    pub fn new(
        board: &'a Board,
        data: &LocalData,
        tt_mv: Option<Move>,
        skip_quiets: bool,
        counter_hist: Option<&PieceHistory>,
        followup_hist: Option<&PieceHistory>,
    ) -> Self {
        let mut moves = Vec::with_capacity(64);

        let mut piece_moves = ArrayVec::<_, 32>::new();

        board.generate_moves(|mvs| {
            piece_moves.push(mvs);
            false
        });

        let opp = board.colors(!board.side_to_move());

        for &(mut mvs) in &piece_moves {
            if skip_quiets {
                mvs.to &= opp;
            }
            for mv in mvs {
                let mut score = match tt_mv {
                    Some(tt_mv) if mv == tt_mv => 1_000_000,
                    _ if opp.has(mv.to) => 100_000 + board.piece_on(mv.to).unwrap() as i32,
                    _ => {
                        data.history.get(board, mv) as i32
                            + counter_hist.map_or(0, |table| table.get(board, mv) as i32)
                            + followup_hist.map_or(0, |table| table.get(board, mv) as i32)
                    }
                };
                match mv.promotion {
                    Some(Piece::Knight) => score -= 400_000,
                    Some(Piece::Rook) => score -= 500_000,
                    Some(Piece::Bishop) => score -= 600_000,
                    _ => {}
                }
                moves.push((mv, score));
            }
        }

        MovePicker {
            _skip_quiets: skip_quiets,
            _board: board,
            moves,
            next_idx: 0,
            has_moves: !piece_moves.is_empty(),
        }
    }

    pub fn next(&mut self, _data: &LocalData) -> Option<(usize, Move, i32)> {
        if self.next_idx >= self.moves.len() {
            return None;
        }

        let i = self.next_idx;
        self.next_idx += 1;

        let mut best = i;
        for j in i + 1..self.moves.len() {
            if self.moves[j].1 > self.moves[best].1 {
                best = j;
            }
        }

        self.moves.swap(i, best);

        Some((i, self.moves[i].0, self.moves[i].1))
    }

    pub fn failed(&self) -> impl Iterator<Item = Move> + '_ {
        self.moves[..self.next_idx - 1].iter().map(|&(mv, _)| mv)
    }

    pub fn has_moves(&self) -> bool {
        self.has_moves
    }
}
