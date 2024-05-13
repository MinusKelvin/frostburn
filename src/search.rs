use core::sync::atomic::Ordering;

use arrayvec::ArrayVec;

use crate::{Search, SearchInfo, MAX_DEPTH};

impl Search<'_> {
    pub fn search(&mut self) {
        let mut score = 0;
        let mut pv = ArrayVec::new();
        let mut depth = 0;
        self.data.on_first_depth = true;

        for new_depth in 1..MAX_DEPTH {
            let result = self.negamax(self.root, new_depth, 0);
            self.data.on_first_depth = false;
            if let Some(new_score) = result {
                score = new_score;
                pv = self.data.pv_table[0].clone();
                depth = new_depth;
            }

            if self.limits.depth.is_some_and(|d| d == depth) {
                self.shared.abort.store(true, Ordering::SeqCst);
            }

            let finished = result.is_none() || self.count_node_and_check_abort(true).is_none();

            let info = SearchInfo {
                depth,
                score,
                nodes: self.shared.nodes.load(Ordering::SeqCst),
                time: (self.clock)(),
                pv: &pv,
                finished,
            };
            (self.info)(info);

            if finished {
                break;
            }
        }
    }
}
