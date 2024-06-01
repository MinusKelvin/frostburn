use arrayvec::ArrayVec;
use cozy_chess::{Board, Move, PieceMoves};

use crate::LocalData;

pub struct MovePicker<'a> {
    skip_quiets: bool,
    has_moves: bool,
    generated: bool,
    board: &'a Board,
    moves: Vec<(Move, i32)>,
    quiets: ArrayVec<PieceMoves, 32>,
    tt_mv: Option<Move>,
    next_idx: usize,
}

impl<'a> MovePicker<'a> {
    pub fn new(board: &'a Board, tt_mv: Option<Move>, skip_quiets: bool) -> Self {
        let opp = board.colors(!board.side_to_move());

        MovePicker {
            skip_quiets,
            generated: false,
            board,
            moves: vec![],
            quiets: ArrayVec::new(),
            tt_mv: tt_mv.filter(|&mv| !skip_quiets || opp.has(mv.to)),
            next_idx: 0,
            has_moves: tt_mv.is_some(),
        }
    }

    fn generate(&mut self, _data: &LocalData) {
        self.moves.reserve(64);
        if let Some(tt_mv) = self.tt_mv {
            self.moves.push((tt_mv, 1_000_000));
        }
        let opp = self.board.colors(!self.board.side_to_move());

        self.board.generate_moves(|mvs| {
            self.has_moves = true;

            let mut captures = mvs;
            captures.to &= opp;

            for mv in captures {
                if Some(mv) == self.tt_mv {
                    continue;
                }
                let victim = self.board.piece_on(mv.to).unwrap();
                self.moves.push((mv, 100_000 + victim as i32));
            }

            if !self.skip_quiets {
                let mut quiets = mvs;
                quiets.to &= !opp;
                if !quiets.is_empty() {
                    self.quiets.push(quiets);
                }
            }

            false
        });

        self.generated = true;
    }

    fn expand_quiets(&mut self, data: &LocalData) {
        for mvs in self.quiets.drain(..) {
            for mv in mvs {
                if Some(mv) == self.tt_mv {
                    continue;
                }
                let score = data.history.get(self.board, mv) as i32;
                self.moves.push((mv, score));
            }
        }
    }

    pub fn next(&mut self, data: &LocalData) -> Option<(usize, Move, i32)> {
        if let Some(tt_mv) = self.tt_mv {
            if self.next_idx == 0 {
                self.next_idx += 1;
                return Some((0, tt_mv, 1_000_000));
            }
        }
        if !self.generated {
            self.generate(data);
        }

        if self.next_idx >= self.moves.len() {
            self.expand_quiets(data);
            if self.next_idx >= self.moves.len() {
                return None;
            }
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

    pub fn has_moves(&mut self) -> bool {
        if !self.has_moves && !self.generated {
            return self.board.generate_moves(|_| true);
        }
        self.has_moves
    }
}
