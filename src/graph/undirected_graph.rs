use super::*;
use haversine::haversine_m_fpair_ord;
use ordered_float::OrderedFloat;
use rayon::prelude::ParallelIterator;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::fmt::Debug;

use itertools::Itertools;
use smallvec::smallvec;
use std::iter;

type SmallVecIntermediates<V> = SmallVec<[V; 1]>;

pub struct UndirectedAdjGraph<V, E> {
    edges: BTreeMap<V, BTreeMap<V, (E, SmallVecIntermediates<V>)>>,
}

impl<V, E> Default for UndirectedAdjGraph<V, E>
where
    V: std::hash::Hash + Eq + Copy + Ord + Send + std::fmt::Debug + Default,
    E: Copy
        + PartialOrd
        + Clone
        + std::fmt::Debug
        + std::ops::Add<Output = E>
        + std::cmp::PartialEq
        + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<V, E> UndirectedAdjGraph<V, E>
where
    V: std::hash::Hash + Eq + Copy + Ord + Send + std::fmt::Debug + Default,
    E: Copy
        + PartialOrd
        + Clone
        + std::fmt::Debug
        + std::ops::Add<Output = E>
        + std::cmp::PartialEq
        + Default,
{
    pub fn new() -> Self {
        Self {
            edges: Default::default(),
        }
    }

    pub fn set(&mut self, i: &V, j: &V, val: E) {
        self.edges
            .entry(*i)
            .or_default()
            .insert(*j, (val, Default::default()));
        self.edges
            .entry(*j)
            .or_default()
            .insert(*i, (val, Default::default()));
    }

    pub fn remove_vertex(&mut self, v: &V) {
        while let Some((other_v, (_weight, _intermediaters))) =
            self.edges.get_mut(v).unwrap().pop_last()
        {
            self.edges.get_mut(&other_v).unwrap().remove(v);
        }
        assert!(self.edges[v].is_empty());
        self.edges.remove(v);
    }

    pub fn get(&self, i: &V, j: &V) -> Option<&E> {
        self.edges
            .get(i)
            .and_then(|from_i| from_i.get(j).map(|(e, _intermediates)| e))
    }

    pub fn get_all(&self, i: &V, j: &V) -> Option<&(E, SmallVecIntermediates<V>)> {
        self.edges.get(i).and_then(|from_i| from_i.get(j))
    }

    pub fn get_intermediates(&self, i: &V, j: &V) -> Option<&[V]> {
        self.get_all(i, j)
            .map(|(_e, intermediates)| intermediates.as_slice())
    }

    /// Return iterator over all the “contracted” edges.
    /// Each element is 1: the edge weight for that “segment”, and 2: an iterator over the vertexes
    /// of this segment (in order)
    pub fn get_all_contracted_edges(&self) -> impl Iterator<Item = (&E, impl Iterator<Item = &V>)> {
        self.edges
            .iter()
            .flat_map(|(a, from_a)| {
                from_a
                    .iter()
                    // since edges are stored twice, only take the first one
                    .filter(move |(b, _)| a < b)
                    .map(move |(b, (edge_weight, inter))| ((a, inter, b), edge_weight))
            })
            .map(|((a, inter, b), edge_weight)| {
                (
                    edge_weight,
                    iter::once(a).chain(inter.iter()).chain(iter::once(b)),
                )
            })
    }

    /// returns each vertex id and how many neighbours it has
    pub fn iter_vertexes_num_neighbours(&self) -> impl Iterator<Item = (&V, usize)> {
        self.edges.iter().map(|(vid, edges)| (vid, edges.len()))
    }

    pub fn contains_vertex(&self, v: &V) -> bool {
        self.edges.contains_key(v)
    }

    /// All the neighbours of this vertex and the edge weight
    pub fn neighbors(&self, i: &V) -> impl Iterator<Item = (&V, &E)> + use<'_, V, E> {
        self.edges[i]
            .iter()
            .map(|(j, (edge_weight, _intermediates))| (j, edge_weight))
    }

    /// Number of neighbours for this vertex.
    pub fn num_neighbors(&self, i: &V) -> usize {
        self.edges.get(i).map_or(0, |es| es.len())
    }

    pub fn len(&self) -> usize {
        self.edges
            .values()
            .map(|from_i| from_i.len())
            .sum::<usize>()
            / 2
    }

    pub fn num_edges(&self) -> usize {
        self.edges
            .values()
            .map(|from_i| from_i.len())
            .sum::<usize>()
            / 2
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
    pub fn num_vertexes(&self) -> usize {
        self.edges.len()
    }

    pub fn vertexes(&self) -> impl Iterator<Item = &V> {
        self.edges.keys()
    }

    pub fn remove_edge(&mut self, i: &V, j: &V) {
        if let Some(from_i) = self.edges.get_mut(i) {
            from_i.remove(j);
            if from_i.is_empty() {
                self.edges.remove(i);
            }
        }
        if let Some(from_j) = self.edges.get_mut(j) {
            from_j.remove(i);
            if from_j.is_empty() {
                self.edges.remove(j);
            }
        }
    }
}

pub type SmallNidVec = SmallVec<[i64; 1]>;
pub type SmallI32Vec = SmallVec<[i32; 1]>;

#[derive(Default, Debug)]
pub struct Graph2 {
    // key is vertex id
    edges: BTreeMap<i64, SmallNidVec>,
}

impl Graph2 {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add_edge(&mut self, vertex1: i64, vertex2: i64) -> bool {
        if vertex1 == vertex2 {
            return false;
        }
        match self.edges.get_mut(&vertex1) {
            Some(others) => {
                if others.contains(&vertex2) {
                    true
                } else {
                    others.push(vertex2);
                    self.edges.entry(vertex2).or_default().push(vertex1);
                    false
                }
            }
            _ => {
                self.edges.insert(vertex1, smallvec![vertex2]);
                self.edges.entry(vertex2).or_default().push(vertex1);
                false
            }
        }
    }

    pub fn vertexes(&self) -> impl Iterator<Item = &i64> {
        self.edges.keys()
    }
    pub fn vertexes_par_iter(&self) -> impl ParallelIterator<Item = &i64> {
        self.edges.par_iter().map(|(k, _v)| k)
    }

    pub fn vertexes_w_num_neighbours(&self) -> impl Iterator<Item = (&i64, usize)> {
        self.edges.iter().map(|(nid, neigh)| (nid, neigh.len()))
    }

    pub fn edges_iter(&self) -> impl Iterator<Item = (&i64, &i64)> {
        self.edges.iter().flat_map(|(nid, neighs)| {
            neighs
                .iter()
                .filter(|other| *other > nid)
                .map(move |other| (nid, other))
        })
    }
    pub fn edges_par_iter(&self) -> impl ParallelIterator<Item = (&i64, &i64)> {
        self.edges.par_iter().flat_map(|(nid, neighs)| {
            neighs
                .par_iter()
                .filter(|other| *other > nid)
                .map(move |other| (nid, other))
        })
    }

    pub fn neighbours(&self, vertex: &i64) -> impl Iterator<Item = &i64> + use<'_> {
        self.edges.get(vertex).into_iter().flat_map(|ns| ns.iter())
    }

    pub fn add_edge_chain(&mut self, vertexes: &[i64]) -> bool {
        // This could be speed up by keeping a reference to the current vertex edges and walking
        // along. saved looking up the same vertex twice in the btreemap
        if vertexes.len() < 2 {
            return false;
        }
        let mut added = false;
        for w in vertexes.windows(2) {
            added |= self.add_edge(w[0], w[1]);
        }
        added
    }

    pub fn num_vertexes(&self) -> usize {
        self.edges.len()
    }
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn first_vertex(&self) -> Option<&i64> {
        self.edges.first_key_value().map(|(k, _v)| k)
    }
    pub fn contains_vertex(&self, vertex: i64) -> bool {
        self.edges.contains_key(&vertex)
    }
    pub fn contains_edge(&self, v1: i64, v2: i64) -> bool {
        self.edges
            .get(&v1)
            .is_some_and(|neighs| neighs.contains(&v2))
    }

    pub fn remove_vertex(&mut self, vertex: i64) -> Option<SmallNidVec> {
        let others = self.edges.remove(&vertex);
        if let Some(ref others) = others {
            for o in others {
                let oo = self.edges.get_mut(o).unwrap();
                oo.retain(|n| *n != vertex);
                if oo.is_empty() {
                    self.edges.remove(o);
                }
            }
        }

        others
    }
    pub fn remove_edge(&mut self, vertex1: i64, vertex2: i64) -> Option<(i64, i64)> {
        if self
            .edges
            .get(&vertex1)
            .is_some_and(|v1_neighs| v1_neighs.contains(&vertex2))
        {
            self.edges
                .get_mut(&vertex1)
                .unwrap()
                .retain(|n| *n != vertex2);
            self.edges
                .get_mut(&vertex2)
                .unwrap()
                .retain(|n| *n != vertex1);
            if self.edges.get(&vertex1).unwrap().is_empty() {
                self.edges.remove(&vertex1);
            }
            if self.edges.get(&vertex2).unwrap().is_empty() {
                self.edges.remove(&vertex2);
            }

            Some((vertex1, vertex2))
        } else {
            None
        }
    }

    pub fn into_disconnected_graphs(
        mut self,
        progress_bar: impl Into<Option<ProgressBar>>,
    ) -> impl Iterator<Item = Self> {
        let mut frontier = Vec::new();
        let progress_bar: Option<ProgressBar> = progress_bar.into();

        std::iter::from_fn(move || {
            if self.is_empty() {
                return None;
            }

            let mut new_graph = Graph2::new();

            frontier.truncate(0);
            frontier.push(*self.first_vertex().unwrap());

            while let Some(nid) = frontier.pop() {
                // might have been already removed when removing another
                if let Some(others) = self.remove_vertex(nid) {
                    if let Some(progress_bar) = &progress_bar {
                        progress_bar.inc(1);
                    }
                    for o in others {
                        if !new_graph.contains_vertex(o) {
                            frontier.push(o);
                        }
                        new_graph.add_edge(nid, o);
                    }
                }
            }

            Some(new_graph)
        })
    }

    pub fn into_lines_random(self) -> impl Iterator<Item = Box<[i64]>> {
        let mut graph = self;

        std::iter::from_fn(move || {
            if graph.is_empty() {
                return None;
            }

            let mut curr_path: Vec<i64> = Vec::new();
            // Try to start “at an end”, not in the middle.
            // i.e. look for a point with only 1 neighbour.
            // Only check a few vertexes, to prevent long generation time.
            // This results in nicer (& smaller) data.
            curr_path.push(
                graph
                    .vertexes_w_num_neighbours()
                    .take(100)
                    .filter(|(_nid, nn)| *nn == 1)
                    .map(|(nid, _)| *nid)
                    .next()
                    .unwrap_or_else(|| *graph.first_vertex().unwrap()),
            );
            let mut next: Option<i64>;
            let mut last: &i64;

            loop {
                last = curr_path.last().unwrap();
                next = graph
                    .neighbours(last)
                    .find(|v| !curr_path.contains(*v))
                    .copied();
                match next {
                    None => {
                        break;
                    }
                    Some(next) => {
                        graph.remove_edge(*last, next).unwrap();
                        curr_path.push(next);
                    }
                }
            }

            Some(curr_path.into_boxed_slice())
        })
    }

    pub fn into_lines_as_crow_flies(
        self,
        nodeid_pos: &impl NodeIdPosition,
    ) -> impl Iterator<Item = Box<[i64]>> + '_ {
        let mut graphs = vec![self];

        std::iter::from_fn(move || {
            if graphs.is_empty() {
                return None;
            }
            let mut graph = graphs.pop().unwrap();
            assert!(!graph.is_empty());

            let target_pair = graph
                .edges
                .keys()
                .par_bridge()
                .map(|n1| (n1, nodeid_pos.get(n1).unwrap()))
                .flat_map(|(n1, p1)| {
                    graph
                        .edges
                        .keys()
                        .par_bridge()
                        .filter(|n2| *n2 > n1)
                        .map(move |n2| ((n1, p1), (n2, nodeid_pos.get(n2).unwrap())))
                })
                .max_by_key(|((_n1, p1), (_n2, p2))| haversine_m_fpair_ord(*p1, *p2));
            let target_pair = target_pair.unwrap();

            let src = target_pair.0;
            let dest = target_pair.1;
            let dest_nid = dest.0;
            let dest_p = dest.1;

            let res = dij::path_one_to_one(
                (*src.0, (OrderedFloat(src.1.0), OrderedFloat(src.1.1))),
                (*dest_nid, (OrderedFloat(dest_p.0), OrderedFloat(dest_p.1))),
                nodeid_pos,
                &graph,
            );
            let path = res.1;

            for edge in path.iter().tuple_windows::<(_, _)>() {
                graph.remove_edge(*edge.0, *edge.1).unwrap();
            }

            if !graph.is_empty() {
                let _old_num_graphs = graphs.len();
                graphs.extend(graph.into_disconnected_graphs(None));
            }

            Some(path.into_boxed_slice())
        })
    }
}
