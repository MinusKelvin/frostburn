use std::ops::{Index, IndexMut};

use bytemuck::{Pod, Zeroable};
use cozy_chess::{Board, Color, Move, Piece, Square};

const MAX_HISTORY: i32 = 1 << 14;

#[derive(Zeroable)]
pub struct PieceHistory {
    table: C<P<Sq<i16>>>,
}

impl PieceHistory {
    pub fn new() -> Self {
        Zeroable::zeroed()
    }

    pub fn get(&self, board: &Board, mv: Move) -> i16 {
        let color = board.side_to_move();
        let piece = board.piece_on(mv.from).unwrap();
        self.table[color][piece][mv.to]
    }

    pub fn update(&mut self, board: &Board, mv: Move, bonus: i16) {
        let color = board.side_to_move();
        let piece = board.piece_on(mv.from).unwrap();
        let slot = &mut self.table[color][piece][mv.to];

        *slot += bonus - (bonus.abs() as i32 * *slot as i32 / MAX_HISTORY) as i16;
    }
}

pub struct ContinuationHistory {
    table: Box<P<Sq<PieceHistory>>>,
}

impl ContinuationHistory {
    pub fn new() -> Self {
        ContinuationHistory {
            table: bytemuck::zeroed_box(),
        }
    }

    pub fn get(&self, prior: Option<(Move, Piece)>) -> Option<&PieceHistory> {
        prior.map(|(mv, piece)| &self.table[piece][mv.to])
    }

    pub fn get_mut(&mut self, prior: Option<(Move, Piece)>) -> Option<&mut PieceHistory> {
        prior.map(|(mv, piece)| &mut self.table[piece][mv.to])
    }
}

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(transparent)]
struct P<T>([T; Piece::NUM]);

impl<T> Index<Piece> for P<T> {
    type Output = T;

    fn index(&self, index: Piece) -> &T {
        &self.0[index as usize]
    }
}

impl<T> IndexMut<Piece> for P<T> {
    fn index_mut(&mut self, index: Piece) -> &mut T {
        &mut self.0[index as usize]
    }
}

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(transparent)]
struct C<T>([T; Color::NUM]);

impl<T> Index<Color> for C<T> {
    type Output = T;

    fn index(&self, index: Color) -> &T {
        &self.0[index as usize]
    }
}

impl<T> IndexMut<Color> for C<T> {
    fn index_mut(&mut self, index: Color) -> &mut T {
        &mut self.0[index as usize]
    }
}

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(transparent)]
struct Sq<T>([T; Square::NUM]);

impl<T> Index<Square> for Sq<T> {
    type Output = T;

    fn index(&self, index: Square) -> &T {
        &self.0[index as usize]
    }
}

impl<T> IndexMut<Square> for Sq<T> {
    fn index_mut(&mut self, index: Square) -> &mut T {
        &mut self.0[index as usize]
    }
}
