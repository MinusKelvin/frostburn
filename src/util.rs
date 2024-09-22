use cozy_chess::{Board, Move, Piece};

pub trait BoardExt {
    fn victim(&self, mv: Move) -> Option<Piece>;
}

impl BoardExt for Board {
    fn victim(&self, mv: Move) -> Option<Piece> {
        let opp_occ = self.colors(!self.side_to_move());
        if opp_occ.has(mv.to) {
            return self.piece_on(mv.to);
        }
        if self.piece_on(mv.from) == Some(Piece::Pawn) && mv.from.file() != mv.to.file() {
            return Some(Piece::Pawn);
        }
        None
    }
}
