use arrayvec::ArrayVec;
use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves, BitBoard,
    Board, Color, Move, Piece, Square,
};

use crate::history::PieceHistory;
use crate::LocalData;

pub struct MovePicker<'a> {
    _skip_quiets: bool,
    has_moves: bool,
    _board: &'a Board,
    moves: Vec<ScoredMove>,
    next_idx: usize,
}

pub struct ScoredMove {
    pub mv: Move,
    pub score: i32,
    pub history: i32,
}

impl<'a> MovePicker<'a> {
    pub fn new(
        board: &'a Board,
        data: &LocalData,
        tt_mv: Option<Move>,
        excluded: Option<Move>,
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
                if excluded.is_some_and(|excluded| excluded == mv) {
                    continue;
                }
                let mut history = 0;
                let mut score = match tt_mv {
                    Some(tt_mv) if mv == tt_mv => 1_000_000,
                    _ if opp.has(mv.to) => {
                        let base = board.piece_on(mv.to).unwrap() as i32;
                        if see_ge(board, mv, 1) {
                            base + 150_000
                        } else {
                            base + 100_000
                        }
                    }
                    _ => {
                        history = data.history.get(board, mv) as i32
                            + counter_hist.map_or(0, |table| table.get(board, mv) as i32)
                            + followup_hist.map_or(0, |table| table.get(board, mv) as i32);
                        history
                    }
                };
                match mv.promotion {
                    Some(Piece::Knight) => score -= 400_000,
                    Some(Piece::Rook) => score -= 500_000,
                    Some(Piece::Bishop) => score -= 600_000,
                    _ => {}
                }
                moves.push(ScoredMove {
                    mv,
                    score,
                    history,
                });
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

    #[inline(always)]
    pub fn next(&mut self, _data: &LocalData) -> Option<(usize, &ScoredMove)> {
        let i = self.next_idx;
        let (best, _) = self.moves
            .iter()
            .enumerate()
            .skip(i)
            .min_by_key(|&(_, mv)| std::cmp::Reverse(mv))?;
        self.moves.swap(i, best);
        self.next_idx += 1;
        Some((i, &self.moves[i]))
    }

    pub fn failed(&self) -> impl Iterator<Item = &ScoredMove> + '_ {
        self.moves[..self.next_idx - 1].iter()
    }

    pub fn has_moves(&self) -> bool {
        self.has_moves
    }
}

pub fn see_ge(pos: &Board, mv: Move, threshold: i32) -> bool {
    const VALUES: [i32; 6] = [10, 30, 33, 50, 90, 0];

    let sq = mv.to;

    let mut gain = pos.piece_on(sq).map_or(0, |p| VALUES[p as usize]) - threshold;

    if gain < 0 {
        // Even in the best case, we fail to beat the threshold
        return false;
    }

    gain -= VALUES[pos.piece_on(mv.from).unwrap() as usize];

    if gain >= 0 {
        // Even if we lose the piece, we beat the threshold
        return true;
    }

    // Track of occupancy and stm changes throughout the exchange
    let mut occupied = pos.occupied() - mv.from.bitboard();
    let mut stm = pos.side_to_move();

    loop {
        // swap side
        gain = -gain - 1;
        stm = !stm;

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

        let Some(attacker) = attacker else {
            // Out of attackers, so stm loses the exchange
            return stm != pos.side_to_move();
        };

        // Perform recapture
        occupied -= attacker.bitboard();
        // Assume we lose the piece we recaptured with
        gain -= VALUES[pos.piece_on(attacker).unwrap() as usize];

        if gain >= 0 {
            // STM wins the exchange
            return stm == pos.side_to_move();
        }
    }
}

impl Ord for ScoredMove {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

impl PartialOrd for ScoredMove {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ScoredMove {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredMove {}
