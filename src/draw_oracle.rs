use cozy_chess::{BitBoard, Board, Color, Piece};

pub fn draw_oracle(pos: &Board) -> bool {
    // The existence of pawns, rooks, or queens is strong evidence against a draw.
    if !pos.pieces(Piece::Pawn).is_empty()
        || !pos.pieces(Piece::Rook).is_empty()
        || !pos.pieces(Piece::Queen).is_empty()
    {
        return false;
    }

    let white_count = pos.colors(Color::White).len();
    let black_count = pos.colors(Color::Black).len();

    let side_with_material = match (white_count, black_count) {
        // bail if both sides have material
        (2.., 2..) => return false,
        (2.., _) => Color::White,
        _ => Color::Black,
    };

    let knights = pos.colored_pieces(side_with_material, Piece::Knight).len();
    let light_bishops =
        (pos.colored_pieces(side_with_material, Piece::Bishop) & BitBoard::LIGHT_SQUARES).len();
    let dark_bishops =
        (pos.colored_pieces(side_with_material, Piece::Bishop) & BitBoard::DARK_SQUARES).len();

    match (knights, light_bishops, dark_bishops) {
        // only one color of bishop is insufficient
        (0, 0, _) => true,
        (0, _, 0) => true,
        // both is potentially winnable
        (0, 1.., 1..) => false,

        // only one knight is insufficient
        (1, 0, 0) => true,
        // two knights is insufficient unless the side-to-lose king is on the edge
        (2, 0, 0) => !BitBoard::EDGES.has(pos.king(!side_with_material)),
        // knight with help is potentially winnable
        (1.., _, _) => false,
    }
}
