use cozy_chess::Board;

pub struct RepetitionTable {
    list: Vec<u64>,
    table: [u8; 4096],
}

impl RepetitionTable {
    pub fn new() -> Self {
        RepetitionTable {
            list: Vec::with_capacity(64),
            table: [0; 4096],
        }
    }

    pub fn push(&mut self, hash: u64) {
        self.list.push(hash);
        self.table[hash as usize % self.table.len()] += 1;
    }

    pub fn pop(&mut self) {
        let hash = self.list.pop().unwrap();
        self.table[hash as usize % self.table.len()] -= 1;
    }

    pub fn is_rep(&self, board: &Board) -> bool {
        let hash = board.hash();
        if self.table[hash as usize % self.table.len()] == 0 {
            return false;
        }

        let to_check = self.list.len().min(board.halfmove_clock() as usize);
        self.list.iter().rev().take(to_check).any(|&h| h == hash)
    }

    pub fn clear(&mut self) {
        self.list.clear();
        self.table = [0; 4096];
    }
}
