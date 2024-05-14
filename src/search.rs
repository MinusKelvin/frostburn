use core::sync::atomic::Ordering;

use arrayvec::ArrayVec;

use crate::{Search, SearchInfo, MAX_DEPTH};

impl Search<'_> {
    pub fn search(&mut self) {
        let mut score = 0;
        let mut pv = ArrayVec::new();
        let mut depth = 0;
        self.data.on_first_depth = true;

        for new_depth in 1.. {
            let result = self.negamax(self.root, -30_000, 30_000, new_depth, 0);
            self.data.on_first_depth = false;
            if let Some(new_score) = result {
                score = new_score;
                pv = self.data.pv_table[0].clone();
                depth = new_depth;
            }

            let mut finished = result.is_none() || self.count_node_and_check_abort(true).is_none();
            let nodes = self.shared.nodes.load(Ordering::SeqCst);

            if depth == MAX_DEPTH
                || self.limits.depth.is_some_and(|d| d == depth)
                || self.limits.min_nodes.is_some_and(|n| nodes >= n)
            {
                self.shared.abort.store(true, Ordering::SeqCst);
                finished = true;
            }

            let info = SearchInfo {
                depth,
                score,
                nodes,
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
