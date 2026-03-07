use super::*;
use crate::inter_store::InterStore;
use haversine::haversine_m_fpair_ord;
use ordered_float::OrderedFloat;
use rand::prelude::*;
use rayon::prelude::ParallelIterator;
use smallvec::SmallVec;
use sorted_slice_store::SortedSliceMap;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fmt::Debug;
use utils::min_max;

use kiddo::{KdTree, SquaredEuclidean};

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
    #[must_use]
    pub fn new() -> Self {
        Self {
            edges: BTreeMap::default(),
        }
    }

    pub fn set(&mut self, i: &V, j: &V, val: E) {
        self.edges
            .entry(*i)
            .or_default()
            .insert(*j, (val, SmallVec::default()));
        self.edges
            .entry(*j)
            .or_default()
            .insert(*i, (val, SmallVec::default()));
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
        self.edges.get(i).map_or(0, std::collections::BTreeMap::len)
    }

    pub fn len(&self) -> usize {
        self.edges
            .values()
            .map(std::collections::BTreeMap::len)
            .sum::<usize>()
            / 2
    }

    pub fn num_edges(&self) -> usize {
        self.edges
            .values()
            .map(std::collections::BTreeMap::len)
            .sum::<usize>()
            / 2
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
    #[must_use]
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

#[derive(Default, Debug, Clone)]
pub struct Graph2 {
    // key is vertex id
    edges: BTreeMap<i64, SmallNidVec>,
}

impl Graph2 {
    #[must_use]
    pub fn new() -> Self {
        Graph2::default()
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

    #[must_use]
    pub fn vertexes(&self) -> impl ExactSizeIterator<Item = &i64> {
        self.edges.keys()
    }
    #[must_use]
    pub fn vertexes_par_iter(&self) -> impl ParallelIterator<Item = &i64> {
        self.edges.par_iter().map(|(k, _v)| k)
    }

    pub fn vertexes_w_num_neighbours(&self) -> impl Iterator<Item = (&i64, usize)> {
        self.edges.iter().map(|(nid, neigh)| (nid, neigh.len()))
    }

    #[must_use]
    pub fn vertexes_w_num_neighbours_par(&self) -> impl ParallelIterator<Item = (&i64, usize)> {
        self.edges.par_iter().map(|(nid, neigh)| (nid, neigh.len()))
    }

    pub fn edges_iter(&self) -> impl Iterator<Item = (&i64, &i64)> {
        self.edges.iter().flat_map(|(nid, neighs)| {
            neighs
                .iter()
                .filter(|other| *other > nid)
                .map(move |other| (nid, other))
        })
    }
    #[must_use]
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

    pub fn num_neighbors(&self, vertex: &i64) -> Option<usize> {
        self.edges.get(vertex).map(smallvec::SmallVec::len)
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

    #[must_use]
    pub fn num_vertexes(&self) -> usize {
        self.edges.len()
    }
    pub fn num_edges(&self) -> usize {
        self.edges
            .values()
            .map(smallvec::SmallVec::len)
            .sum::<usize>()
            / 2
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    #[must_use]
    pub fn first_vertex(&self) -> Option<&i64> {
        self.edges.first_key_value().map(|(k, _v)| k)
    }
    #[must_use]
    pub fn contains_vertex(&self, vertex: i64) -> bool {
        self.edges.contains_key(&vertex)
    }
    #[must_use]
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
                None,
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

    /// Return a random sample of points in this graph, somewhat spread out.
    #[allow(clippy::type_complexity)]
    pub fn random_sample_vertexes(
        &self,
        num: usize,
        nodeid_pos: &impl NodeIdPosition,
    ) -> Box<[i64]> {
        if num >= self.num_vertexes() {
            let all_nodes = self.vertexes().copied().collect::<Vec<_>>();
            return all_nodes.into_boxed_slice();
        }

        let all_nodes = self
            .vertexes()
            .copied()
            .collect::<Vec<i64>>()
            .into_boxed_slice();
        assert!(!all_nodes.is_empty());

        // In each iteration, how many to
        let k = 100;

        // Need to quickly check existing nodes, so keep as a hashmap
        let mut new_nodes = HashSet::with_capacity(num);

        let mut kdtree: KdTree<f64, 2> = KdTree::with_capacity(num);
        let mut rng = &mut rand::rng();

        let first = *all_nodes.choose(&mut rng).unwrap();
        let pos = nodeid_pos.get_arr(&first).unwrap();
        new_nodes.insert(first);
        kdtree.add(&pos, first.try_into().unwrap());

        // Buffer of possible nodes for each iteration.
        let mut possible_nodes = Vec::with_capacity(k);

        while new_nodes.len() < num {
            // We need to exclude nodes we
            // It's quicker to repidly call choose_multiple rather than multiple .choose.
            possible_nodes.truncate(0);
            while possible_nodes.len() < k {
                possible_nodes.extend(
                    all_nodes
                        .sample(&mut rng, k - possible_nodes.len() + 1)
                        .filter(|nid| !new_nodes.contains(nid))
                        .map(|nid| {
                            let pos = nodeid_pos.get_arr(nid).unwrap();
                            let dist = kdtree.nearest_one::<SquaredEuclidean>(&pos).distance;

                            // dist is minus → largest dist is first
                            (OrderedFloat(-dist), *nid, pos)
                        }),
                );
            }

            // Take the node with the largest distance
            possible_nodes.par_sort_by_key(|(dist, _nid, _pos)| *dist);
            let (_dist, nid, pos) = possible_nodes[0];

            // save it
            kdtree.add(&pos, nid.try_into().unwrap());
            new_nodes.insert(nid);
        }

        let new_nodes = new_nodes.into_iter().collect::<Vec<_>>();
        new_nodes.into_boxed_slice()
    }

    pub fn betweenness_centrality(
        &self,
        nodes: &[i64],
        nodeid_pos: &impl NodeIdPosition,
        inter_store: &inter_store::InterStore,
        progress_bar: impl Into<Option<ProgressBar>>,
    ) -> SortedSliceMap<(i64, i64), u64> {
        let progress_bar: Option<ProgressBar> = progress_bar.into();

        let edge_lengths = SortedSliceMap::from_iter(self.edges_iter().map(|(nid1, nid2)| {
            let edge_len = inter_store
                .expand_undirected(*nid1, *nid2)
                .map(|nid| nodeid_pos.get(&nid).unwrap())
                .tuple_windows::<(_, _)>()
                .par_bridge()
                .map(|(p1, p2)| haversine::haversine_m_fpair(p1, p2))
                .sum::<f64>();

            let edge_len = (edge_len * 100.).round() as u64;

            ((*nid1, *nid2), edge_len)
        }));

        // The results of the Betweenness Centrality. key = segment (nid1↭nid2), value = the BC
        // value
        let bc_res =
            SortedSliceMap::from_iter(self.edges_iter().map(|(&nid1, &nid2)| ((nid1, nid2), 0)));
        let bc_res = Arc::new(Mutex::new(bc_res));

        nodes.par_iter().enumerate().for_each_with(
            (bc_res.clone(), HashMap::new()),
            |(bc_res, prev_dist), (i, nid0)| {
                let target_nodes = &nodes[(i + 1)..];

                // ↓ Calculate the shortest path to everywhere
                dij::dij_single(*nid0, self, &edge_lengths, prev_dist);

                // Now generate all the paths
                let mut new_segs: Vec<((i64, i64), u64)> = Vec::with_capacity(target_nodes.len());

                // Keep track of all the
                let mut buf_segs: Vec<(u64, i64, _)> = Vec::with_capacity(target_nodes.len());

                buf_segs.extend(
                    target_nodes
                        .iter()
                        .copied()
                        .map(|nid_n| (prev_dist[&nid_n].1, nid_n, 1)),
                );
                buf_segs.par_sort_by_key(|(dist, nid, _acc)| (*dist, *nid));

                while let Some((_dist, nid_b, acc)) = buf_segs.pop() {
                    if &nid_b == nid0 {
                        continue;
                    }
                    let (nid_a, new_dist) = prev_dist[&nid_b];

                    // save this segment
                    new_segs.push((min_max(nid_a, nid_b), acc));

                    let k = buf_segs.partition_point(|(thisdist, nid, _acc)| {
                        (*thisdist, *nid).le(&(new_dist, nid_a))
                    });
                    if k >= buf_segs.len() {
                        // put it on the end
                        buf_segs.push((new_dist, nid_a, acc));
                    } else if buf_segs[k].1 == nid_a {
                        // we update this item
                        buf_segs[k].2 += acc;
                    } else {
                        // new item, put it here
                        buf_segs.insert(k, (new_dist, nid_a, acc));
                    }
                }

                let mut bc_res = bc_res.lock().unwrap();
                for ((nid_a, nid_b), val) in new_segs.into_iter() {
                    *bc_res.get_mut(&(nid_a, nid_b)).unwrap() += val;
                }
                if let Some(progress_bar) = &progress_bar {
                    progress_bar.inc(target_nodes.len() as u64);
                }
            },
        );

        Arc::try_unwrap(bc_res).unwrap().into_inner().unwrap()
    }

    pub fn compress_graph(
        &mut self,
        inter_store: &mut Arc<Mutex<&mut InterStore>>,
        remove_old_inters: bool,
        never_remove_vertexes: impl Fn(i64) -> bool + Sync,
    ) {
        let num_orig_vertexes = self.num_vertexes();

        // these nodes have 2 neighbours, but shouldn't be removed.
        let mut vertexes_already_excluded = HashSet::new();

        // temp needed buffers
        let mut vertex_queue: Vec<i64> = Vec::new();
        let mut tmp_inters = Vec::new();

        loop {
            vertex_queue.par_extend(self.vertexes_w_num_neighbours_par().filter_map(
                |(nid, nneigh)| {
                    if nneigh == 2
                        && !vertexes_already_excluded.contains(nid)
                        && !never_remove_vertexes(*nid)
                    {
                        Some(*nid)
                    } else {
                        None
                    }
                },
            ));
            if vertex_queue.is_empty() {
                break;
            }

            while let Some(nid) = vertex_queue.pop() {
                if self.num_neighbors(&nid) != Some(2) || never_remove_vertexes(nid) {
                    continue;
                }
                let mut others = self.remove_vertex(nid).unwrap();
                let nid_b = others.pop().unwrap();
                let nid_a = others.pop().unwrap();

                if self.contains_edge(nid_a, nid_b) {
                    // There's already an edge from a ↔ b, so don't another.
                    // we need to undo what we did here
                    self.add_edge(nid, nid_a);
                    self.add_edge(nid, nid_b);

                    // don't look at this node again
                    vertexes_already_excluded.insert(nid);
                    continue;
                }

                let mut inter_store = inter_store.lock().unwrap();
                tmp_inters.truncate(0);
                tmp_inters.extend(inter_store.inters_undirected(&nid_a, &nid));
                tmp_inters.push(nid);
                tmp_inters.extend(inter_store.inters_undirected(&nid, &nid_b));

                if remove_old_inters {
                    inter_store.remove_undirected(&nid_a, &nid);
                    inter_store.remove_undirected(&nid, &nid_b);
                }
                inter_store.insert_undirected((nid_a, nid_b), &tmp_inters);

                self.add_edge(nid_a, nid_b);

                // potential shortcut
                vertex_queue.push(nid_a);
                vertex_queue.push(nid_b);
            }
        }

        debug!(
            "Removed {} vertexes ({}%)",
            num_orig_vertexes - self.num_vertexes(),
            (num_orig_vertexes - self.num_vertexes()) * 100 / num_orig_vertexes,
        );
    }

    pub fn remove_spikes(&mut self, never_remove_vertexes: impl Fn(i64) -> bool + Sync) {
        let num_orig_vertexes = self.num_vertexes();

        // temp needed buffers
        let mut vertex_queue: Vec<i64> = Vec::new();

        loop {
            vertex_queue.par_extend(self.vertexes_w_num_neighbours_par().filter_map(
                |(nid, nneigh)| {
                    if nneigh == 1 && !never_remove_vertexes(*nid) {
                        Some(*nid)
                    } else {
                        None
                    }
                },
            ));
            if vertex_queue.is_empty() {
                break;
            }

            while let Some(nid) = vertex_queue.pop() {
                if self.num_neighbors(&nid) != Some(1) || never_remove_vertexes(nid) {
                    continue;
                }
                let mut others = self.remove_vertex(nid).unwrap();
                let nid_a = others.pop().unwrap();

                vertex_queue.push(nid_a);
            }
        }

        debug!(
            "Removed {} vertexes ({}%)",
            num_orig_vertexes - self.num_vertexes(),
            (num_orig_vertexes - self.num_vertexes()) * 100 / num_orig_vertexes,
        );
    }
}
