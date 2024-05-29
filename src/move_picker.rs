use cozy_chess::{Board, Move};

use crate::LocalData;

pub struct MovePicker<'a> {
    _skip_quiets: bool,
    _board: &'a Board,
    moves: Vec<(Move, i32)>,
    next_idx: usize,
}

impl<'a> MovePicker<'a> {
    pub fn new(board: &'a Board, data: &LocalData, tt_mv: Option<Move>, skip_quiets: bool) -> Self {
        let mut moves = Vec::with_capacity(64);

        board.generate_moves(|mut mvs| {
            if skip_quiets {
                mvs.to &= board.colors(!board.side_to_move());
            }
            for mv in mvs {
                let score = if tt_mv.is_some_and(|tt_mv| mv == tt_mv) {
                    1_000_000
                } else if board.colors(!board.side_to_move()).has(mv.to) {
                    100_000 + board.piece_on(mv.to).unwrap() as i32
                } else {
                    data.history.get(board, mv) as i32
                };
                moves.push((mv, score));
            }
            false
        });

        MovePicker {
            _skip_quiets: skip_quiets,
            _board: board,
            moves,
            next_idx: 0,
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
}
