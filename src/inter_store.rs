//! Store the intermediate nodes

use std::collections::HashMap;
use std::iter;

#[derive(Default, Debug)]
pub struct InterStore(HashMap<(i64, i64), Box<[u8]>>);

impl InterStore {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn contains_undirected(&self, from: &i64, to: &i64) -> bool {
        let (from, to) = min_max(from, to);
        self.0.contains_key(&(*from, *to))
    }
    pub fn contains_directed(&self, from: &i64, to: &i64) -> bool {
        self.0.contains_key(&(*from, *to))
    }

    pub fn insert_directed(&mut self, edge: (i64, i64), nids: &[i64]) {
        let bytes = vartyint::write_many_delta_new(nids).into_boxed_slice();
        self.0.insert(edge, bytes);
    }
    pub fn insert_undirected(&mut self, edge: (i64, i64), nids: &[i64]) {
        let mut nids = nids.to_vec();
        if edge.0 < edge.1 {
            let bytes = vartyint::write_many_delta_new(&nids).into_boxed_slice();
            self.0.insert(edge, bytes);
        } else {
            nids.reverse();
            let bytes = vartyint::write_many_delta_new(&nids).into_boxed_slice();
            self.0.insert((edge.1, edge.0), bytes);
        }
    }
    pub fn inters_directed(
        &self,
        from: &i64,
        to: &i64,
    ) -> impl Iterator<Item = i64> + '_ + use<'_> {
        self.0
            .get(&(*from, *to))
            .into_iter()
            .flat_map(|it_bytes| vartyint::read_many_delta(it_bytes).map(|res| res.unwrap()))
    }

    pub fn inters_undirected(
        &self,
        from: &i64,
        to: &i64,
    ) -> impl Iterator<Item = i64> + '_ + use<'_> {
        if from < to {
            Box::new(
                self.0.get(&(*from, *to)).into_iter().flat_map(|it_bytes| {
                    vartyint::read_many_delta(it_bytes).map(|res| res.unwrap())
                }),
            ) as Box<dyn Iterator<Item = i64>>
        } else {
            let mut inters = self
                .0
                .get(&(*to, *from))
                .into_iter()
                .flat_map(|it_bytes| vartyint::read_many_delta(it_bytes).map(|res| res.unwrap()))
                .collect::<Vec<i64>>();
            inters.reverse();
            Box::new(inters.into_iter()) as Box<dyn Iterator<Item = i64>>
        }
    }

    pub fn expand_directed(&self, from: i64, to: i64) -> impl Iterator<Item = i64> + '_ {
        iter::once(from)
            .chain(self.inters_directed(&from, &to))
            .chain(iter::once(to))
    }
    pub fn expand_undirected(&self, from: i64, to: i64) -> impl Iterator<Item = i64> + '_ {
        let (from, to) = min_max(from, to);
        self.expand_directed(from, to)
    }

    pub fn expand_line_directed<'a>(&'a self, line: &'a [i64]) -> impl Iterator<Item = i64> + 'a {
        iter::once(line[0]).chain(line.windows(2).flat_map(|seg| {
            self.inters_directed(&seg[0], &seg[1])
                .chain(iter::once(seg[1]))
        }))
    }
    pub fn expand_line_undirected<'a>(&'a self, line: &'a [i64]) -> impl Iterator<Item = i64> + 'a {
        iter::once(line[0]).chain(line.windows(2).flat_map(|seg| {
            self.inters_undirected(&seg[0], &seg[1])
                .chain(iter::once(seg[1]))
        }))
    }

    pub fn all_inter_nids(&self) -> impl Iterator<Item = i64> + '_ {
        self.0
            .values()
            .flat_map(|seg_bytes| vartyint::read_many_delta(seg_bytes).map(Result::unwrap))
    }
}

fn min_max<T: PartialOrd>(a: T, b: T) -> (T, T) {
    if a < b { (a, b) } else { (b, a) }
}
