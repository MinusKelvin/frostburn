use core::sync::atomic::Ordering;

use arrayvec::ArrayVec;

use crate::{Eval, Search, SearchInfo, MAX_DEPTH};

impl Search<'_> {
    pub fn search(mut self) {
        let mut score = Eval::cp(0);
        let mut pv = ArrayVec::new();
        let mut depth = 0;
        self.data.on_first_depth = true;

        // simplify history so we can detect 2-fold
        let start =
            self.history.len() - (self.root.halfmove_clock() as usize).min(self.history.len());
        self.history[start..].sort_unstable();
        self.history = self.history[start..]
            .windows(2)
            .filter_map(|s| (s[0] == s[1]).then_some(s[0]))
            .collect();

        self.data.history.decay();
        self.data.counter_hist.decay();

        for new_depth in 1.. {
            let result = self.negamax::<true>(self.root, Eval::mated(0), Eval::mating(0), new_depth, 0);
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
