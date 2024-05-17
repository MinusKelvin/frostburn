use cozy_chess::{Board, Color, Move, Piece, Square};

const MAX_HISTORY: i32 = 1 << 14;

pub struct PieceHistory {
    table: [[[i16; Square::NUM]; Piece::NUM]; Color::NUM],
}

impl PieceHistory {
    pub fn new() -> Self {
        PieceHistory {
            table: [[[0; Square::NUM]; Piece::NUM]; Color::NUM],
        }
    }

    pub fn get(&self, board: &Board, mv: Move) -> i16 {
        let color = board.side_to_move();
        let piece = board.piece_on(mv.from).unwrap();
        self.table[color as usize][piece as usize][mv.to as usize]
    }

    pub fn update(&mut self, board: &Board, mv: Move, bonus: i16) {
        let color = board.side_to_move();
        let piece = board.piece_on(mv.from).unwrap();
        let slot = &mut self.table[color as usize][piece as usize][mv.to as usize];

        *slot += bonus - (bonus.abs() as i32 * *slot as i32 / MAX_HISTORY) as i16;
    }
}
