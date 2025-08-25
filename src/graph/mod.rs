use super::*;
use haversine::haversine_m_fpair_ord;
use log::warn;
use ordered_float::OrderedFloat;
use rayon::prelude::ParallelIterator;
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashSet};
use std::fmt::Debug;

use crate::kosaraju;
use itertools::Itertools;
use smallvec::smallvec;
use std::iter;

type SmallVecIntermediates<V> = SmallVec<[V; 1]>;

pub(crate) struct UndirectedAdjGraph<V, E> {
    edges: BTreeMap<V, BTreeMap<V, (E, SmallVecIntermediates<V>)>>,
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
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            edges: Default::default(),
        }
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn remove_vertex(&mut self, v: &V) {
        while let Some((other_v, (_weight, _intermediaters))) =
            self.edges.get_mut(v).unwrap().pop_last()
        {
            self.edges.get_mut(&other_v).unwrap().remove(v);
        }
        assert!(self.edges[v].is_empty());
        self.edges.remove(v);
    }

    #[allow(unused)]
    pub fn get(&self, i: &V, j: &V) -> Option<&E> {
        self.edges
            .get(i)
            .and_then(|from_i| from_i.get(j).map(|(e, _intermediates)| e))
    }

    #[allow(dead_code)]
    pub fn get_all(&self, i: &V, j: &V) -> Option<&(E, SmallVecIntermediates<V>)> {
        self.edges.get(i).and_then(|from_i| from_i.get(j))
    }

    #[allow(dead_code)]
    pub fn get_intermediates(&self, i: &V, j: &V) -> Option<&[V]> {
        self.get_all(i, j)
            .map(|(_e, intermediates)| intermediates.as_slice())
    }

    #[allow(unused)]
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

    #[allow(dead_code)]
    /// returns each vertex id and how many neighbours it has
    pub fn iter_vertexes_num_neighbours(&self) -> impl Iterator<Item = (&V, usize)> {
        self.edges.iter().map(|(vid, edges)| (vid, edges.len()))
    }

    #[allow(dead_code)]
    pub fn contains_vertex(&self, v: &V) -> bool {
        self.edges.contains_key(v)
    }

    /// All the neighbours of this vertex and the edge weight
    pub fn neighbors(&self, i: &V) -> impl Iterator<Item = (&V, &E)> + use<'_, V, E> {
        self.edges[i]
            .iter()
            .map(|(j, (edge_weight, _intermediates))| (j, edge_weight))
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn num_edges(&self) -> usize {
        self.edges
            .values()
            .map(|from_i| from_i.len())
            .sum::<usize>()
            / 2
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
    pub fn num_vertexes(&self) -> usize {
        self.edges.len()
    }

    #[allow(dead_code)]
    pub fn vertexes(&self) -> impl Iterator<Item = &V> {
        self.edges.keys()
    }

    #[allow(dead_code)]
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

pub trait DirectedGraphTrait: Send + Sized {
    fn new() -> Self;

    fn in_neighbours(&self, to_vertex: i64) -> impl Iterator<Item = i64>;
    fn num_in_neighbours(&self, to_vertex: i64) -> usize {
        self.in_neighbours(to_vertex).count()
    }
    /// Returns (a, to_vertex) where a is an in neighbour of to_vertex
    fn in_edges(&self, to_vertex: i64) -> impl Iterator<Item = (i64, i64)> {
        self.in_neighbours(to_vertex)
            .map(move |nid0| (nid0, to_vertex))
    }
    fn out_neighbours(&self, from_vertex: i64) -> impl Iterator<Item = i64>;
    fn num_out_neighbours(&self, from_vertex: i64) -> usize {
        self.out_neighbours(from_vertex).count()
    }
    /// Returns (from_vertex, b) where b is an out neighbour of from_vertex
    fn out_edges(&self, from_vertex: i64) -> impl Iterator<Item = (i64, i64)> {
        self.out_neighbours(from_vertex)
            .map(move |nid2| (from_vertex, nid2))
    }

    // For an edge (defined by 2 vertexes), return all other edges which are connected to the first
    // or last vertex (excl. this edge)
    fn all_connected_edges(&self, edge: &(i64, i64)) -> impl Iterator<Item = (i64, i64)> {
        self.in_neighbours(edge.0)
            .map(|v0| (v0, edge.0))
            .chain(self.out_neighbours(edge.0).map(|v2| (edge.0, v2)))
            .chain(self.in_neighbours(edge.1).map(|v0| (v0, edge.1)))
            .chain(self.out_neighbours(edge.1).map(|v2| (edge.1, v2)))
            .filter(move |new_edge| new_edge != edge)
    }

    fn contains_edge(&self, from_vertex: impl Into<i64>, to_vertex: impl Into<i64>) -> bool {
        let to_vertex = to_vertex.into();
        self.out_neighbours(from_vertex.into())
            .any(|v| v == to_vertex)
    }

    fn num_vertexes(&self) -> usize;
    fn num_edges(&self) -> usize {
        self.edges_iter().count()
    }
    /// True iff this vertex is in this graph
    fn contains_vertex(&self, vid: &i64) -> bool;

    /// Iterator over all edges
    fn edges_iter(&self) -> impl Iterator<Item = (i64, i64)> + '_;
    fn edges_par_iter(&self) -> impl ParallelIterator<Item = (i64, i64)>;

    /// returns each vertex and the number of out edges
    fn vertexes_and_num_outs(&self) -> impl Iterator<Item = (i64, usize)> + '_;

    fn len(&self) -> (usize, usize) {
        (self.num_vertexes(), self.num_edges())
    }

    fn is_empty(&self) -> bool {
        self.num_vertexes() == 0
    }

    /// Iterator (in any order) of vertexes which are the destination of an edge
    fn dest_vertexes_jumbled(&self) -> impl ParallelIterator<Item = i64> {
        self.edges_par_iter().map(|(_src, dest)| dest)
    }
    /// Iterator (in any order) of vertexes which are the src of an edge
    fn src_vertexes_jumbled(&self) -> impl Iterator<Item = i64> {
        self.edges_iter().map(|(src, _dest)| src)
    }

    /// True iff this vertex has an outgoing edge
    fn vertex_has_outgoing(&self, vid: &i64) -> bool {
        self.out_neighbours(*vid).next().is_some()
    }

    fn detailed_size(&self) -> String;

    /// True iff this vertex does not have ≥1 edges. this happens with zero in edges, or if the
    /// edge doesn't exist
    /// when doing topological sorting, we remove edges, which can remove the vertex when there are
    /// no more incoming
    fn num_ins_zero(&self, vid: &i64) -> bool {
        self.num_in_neighbours(*vid) == 0
    }

    /// returns each vertex and the number of in & out edges
    fn vertexes_and_num_ins_outs(&self) -> impl Iterator<Item = (i64, usize, usize)> + '_;

    /// Iterator (in any order, possibly with dupes) of vertexes which do not have outgoing edges
    fn vertexes_wo_outgoing_jumbled(&self) -> impl ParallelIterator<Item = i64>
    where
        Self: Sync,
    {
        self.dest_vertexes_jumbled()
            .filter(|v| !self.vertex_has_outgoing(v))
    }

    /// starting at point `nid`, follow all upstreams, in a DFS manner
    fn all_in_edges_recursive(
        &self,
        nid: i64,
        incl_nid: impl Fn(&i64) -> bool,
        nodeid_pos: &impl NodeIdPosition,
    ) -> impl Iterator<Item = Vec<(f64, f64)>> {
        let mut frontier: SmallVec<[_; 1]> = smallvec::smallvec![];
        let mut seen_vertexes = HashSet::new();
        if incl_nid(&nid) {
            frontier.push((nid, None));
        }

        // Somehow in this, nids are getting added to the frontier many times, and this is causing
        // massive duplications of part of nodes

        std::iter::from_fn(move || {
            if frontier.is_empty() {
                return None;
            }

            let (mut curr_point, opt_prev_point) = frontier.pop().unwrap();
            let mut curr_latlng;
            let mut curr_path = Vec::with_capacity(10);

            if let Some(prev_point) = opt_prev_point {
                curr_path.push(prev_point);
            }
            loop {
                curr_latlng = nodeid_pos.get(&curr_point).unwrap();
                curr_path.push(curr_latlng);
                seen_vertexes.insert(curr_point);
                let mut ins = self
                    .in_neighbours(curr_point)
                    .filter(|i| !seen_vertexes.contains(i));
                match ins.next() {
                    Some(nxt) => {
                        // any other out neighbours of this point need to be visited later
                        frontier.extend(
                            ins.filter(|n| incl_nid(n))
                                .map(|in_nid| (in_nid, Some(curr_latlng))),
                        );

                        curr_point = nxt;
                        continue;
                    }
                    _ => {
                        // no more neighbours here
                        curr_path.reverse();
                        return Some(curr_path);
                    }
                }
            }
        })
    }
    /// Vertex v exists, and has either in neighbours, xor out neighbours (but not both)
    fn neighbors_in_xor_out(&mut self, v: &i64) -> bool {
        (self.num_in_neighbours(*v) == 0) ^ (self.num_out_neighbours(*v) == 0)
    }
    fn delete_edge(&mut self, vertex1: &i64, vertex2: &i64);

    fn expand_edge(&self, vertex1: i64, vertex2: i64) -> impl Iterator<Item = i64> + '_ {
        iter::once(vertex1).chain(iter::once(vertex2))
    }

    /// Removes this vertex (& associated edges) from this graph
    fn delete_vertex(&mut self, vertex: &i64);

    fn contract_vertex(&mut self, vertex: &i64, replacement: &i64);
    /// Iterator over all vertexes
    fn vertexes(&self) -> impl Iterator<Item = i64> + '_;

    fn add_edge(&mut self, vertex1: i64, vertex2: i64) -> bool;
    /// Add many vertexes & edges
    fn add_edge_chain(&mut self, vertexes: &[i64]) -> bool {
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

    fn into_vertexes_topologically_sorted(self, sorting_nodes_bar: &ProgressBar) -> Vec<i64> {
        let mut g = self;
        let mut result = Vec::with_capacity(g.num_vertexes());
        let mut frontier: Vec<i64> = Vec::new();

        let mut others = SmallNidVec::new();
        loop {
            frontier.extend(
                g.vertexes_and_num_ins_outs().filter_map(
                    |(v, num_ins, _num_outs)| if num_ins == 0 { Some(v) } else { None },
                ),
            );
            if frontier.is_empty() {
                break;
            }

            while let Some(v) = frontier.pop() {
                result.push(v);
                sorting_nodes_bar.inc(1);

                // have to save to another Vec to prevent lifetimes
                others.truncate(0);
                others.extend(g.out_neighbours(v));
                for other in others.drain(..) {
                    g.delete_edge(&v, &other);
                    if g.num_ins_zero(&other) {
                        frontier.push(other);
                    }
                }
            }
        }

        assert!(
            g.is_empty(),
            "num_vertexes = {} ≠ 0, first remaining edge: {:?}",
            g.num_vertexes(),
            g.edges_iter().next().unwrap()
        );

        result
    }

    fn into_disconnected_graphs(self, progress_bar: &ProgressBar) -> impl Iterator<Item = Self>;

    fn strongly_connected_components(
        &self,
        calc_components_bar: &ProgressBar,
    ) -> Vec<Vec<[i64; 2]>> {
        if self.is_empty() {
            return vec![];
        }
        let component_for_vertex = kosaraju::kosaraju(self, calc_components_bar);

        debug!(
            "kosaraju alg finished. Have {} maps of root vertexes, for {} vertexes",
            component_for_vertex.len(),
            self.num_vertexes(),
        );
        let mut components: HashMap<i64, HashSet<i64>> = HashMap::new();
        for (v, root_v) in component_for_vertex.iter() {
            components
                .entry(*root_v)
                .or_insert_with(|| std::iter::once(*root_v).collect())
                .insert(*v);
        }

        // Convert list of vertexes into linestrings
        components
            .into_values()
            .map(|cycle| {
                cycle
                    .iter()
                    .flat_map(|nid| self.out_neighbours(*nid).map(move |other| [*nid, other]))
                    // Don't include a line segment to an outneighbour which isn't in the cycle. we
                    // alrady know nids[0] is in the cycle
                    .filter(|nids| cycle.contains(&nids[1]))
                    // Expand the intermediate
                    .flat_map(|nids| {
                        self.expand_edge(nids[0], nids[1])
                            .tuple_windows::<(i64, i64)>()
                    })
                    .map(|nid_tuple| [nid_tuple.0, nid_tuple.1])
                    .collect()
            })
            .collect()
    }

    fn delete_vertex_if_unconnected(&mut self, vertex: &i64) {
        if self.num_in_neighbours(*vertex) == 0 && self.num_out_neighbours(*vertex) == 0 {
            self.delete_vertex(vertex);
        }
    }
}

pub trait ContractableDirectedGraph: DirectedGraphTrait {
    fn add_edge_contractable(
        &mut self,
        vertex1: i64,
        vertex2: i64,
        can_contract_vertex: &impl Fn(&i64) -> bool,
    ) -> bool;
    /// Add a sequence of vertexes to this graph, attempting to contract on the go.
    fn add_edge_chain_contractable(
        &mut self,
        vertexes: &[i64],
        can_contract_vertex: &impl Fn(&i64) -> bool,
    ) {
        // add the chain starting from the end, so the contracting can work during adding.
        for (a, b) in rwindows2(vertexes) {
            self.add_edge_contractable(*a, *b, can_contract_vertex);
        }
    }

    fn attempt_contract_vertex(&mut self, vertex: &i64) -> bool;
    fn attempt_contract_all(&mut self, can_contract_vertex: &impl Fn(&i64) -> bool) -> bool;

    fn strongly_connected_components(
        &self,
        calc_components_bar: &ProgressBar,
    ) -> Vec<Vec<[i64; 2]>> {
        if self.is_empty() {
            return vec![];
        }
        let component_for_vertex = kosaraju::kosaraju(self, calc_components_bar);

        debug!(
            "kosaraju alg finished. Have {} maps of root vertexes, for {} vertexes",
            component_for_vertex.len(),
            self.num_vertexes(),
        );
        let mut components: HashMap<i64, HashSet<i64>> = HashMap::new();
        for (v, root_v) in component_for_vertex.iter() {
            components
                .entry(*root_v)
                .or_insert_with(|| std::iter::once(*root_v).collect())
                .insert(*v);
        }

        // Convert list of vertexes into linestrings
        components
            .into_values()
            .map(|cycle| {
                cycle
                    .iter()
                    .flat_map(|nid| self.out_neighbours(*nid).map(move |other| [*nid, other]))
                    // Don't include a line segment to an outneighbour which isn't in the cycle. we
                    // alrady know nids[0] is in the cycle
                    .filter(|nids| cycle.contains(&nids[1]))
                    .collect()
            })
            .collect()
    }
}

/// A graph which stores a list of all incoming and outgoing edges
#[derive(Default, Debug, Clone)]
pub struct DirectedGraph2<V, E>
where
    V: Send + Default + Clone + Sync + Debug,
    E: Send + Default + Clone + Sync + Debug,
{
    // key is vertex id
    // value is ( Vertex properties,
    //            list of vertex idx that come in here, so you can look them up quick
    //            list of outgoing vertexes and the edge property for that edge
    edges: BTreeMap<i64, (V, SmallVec<[i64; 1]>, SmallVec<[(i64, E); 1]>)>,
}

#[allow(dead_code)]
impl<V, E> DirectedGraph2<V, E>
where
    V: Send + Default + Clone + Sync + Debug,
    E: Send + Default + Clone + Sync + Debug,
{
    pub fn vertex_property(&self, vertex: &i64) -> Option<&V> {
        self.edges.get(vertex).map(|v| &v.0)
    }
    pub fn vertex_property_unchecked(&self, vertex: &i64) -> &V {
        self.vertex_property(vertex).unwrap()
    }

    pub fn set_vertex_property(&mut self, vertex: &i64, property: V) {
        self.edges.entry(*vertex).or_default().0 = property;
    }
    pub fn vertex_property_mut(&mut self, vertex: &i64) -> &mut V {
        &mut self.edges.entry(*vertex).or_default().0
    }

    pub fn edge_property(&self, edge: (i64, i64)) -> Option<&E> {
        self.edges
            .get(&edge.0)
            .map(|(_, _, outs)| outs)
            .and_then(|outs| {
                outs.iter()
                    .filter(|(vid, _p)| *vid == edge.1)
                    .next()
                    .map(|(_vid, prop)| prop)
            })
    }
    pub fn edge_property_unchecked(&self, edge: (i64, i64)) -> &E {
        self.edge_property(edge).unwrap()
    }

    pub fn set_edge_property(&mut self, edge: (i64, i64), property: E) {
        if edge.0 == edge.1 {
            return;
        }
        *self.edge_property_mut(edge) = property;
    }

    pub fn edge_property_mut(&mut self, edge: (i64, i64)) -> &mut E {
        let vertex = self.edges.entry(edge.0).or_default();
        let outs = &mut vertex.2;
        if !outs.iter().any(|(oid, _eprop)| *oid == edge.1) {
            outs.push((edge.1, E::default()));
        }

        outs.iter_mut()
            .filter(|(oid, _)| *oid == edge.1)
            .map(|(_, eprop)| eprop)
            .next()
            .unwrap()
    }

    pub fn add_edge_w_prop(&mut self, vertex1: i64, vertex2: i64, eprop: E) {
        if vertex1 == vertex2 {
            return;
        }
        self.add_edge(vertex1, vertex2);
        self.set_edge_property((vertex1, vertex2), eprop);
    }

    pub fn add_vertex_w_prop(&mut self, vertex: i64, vprop: V) {
        // TODO remove all clones
        self.edges
            .entry(vertex)
            .and_modify(|o| o.0 = vprop.clone())
            .or_insert((vprop, Default::default(), Default::default()));
    }

    /// Returns (from_vertex, b, E) where b is an out neighbour of from_vertex
    pub fn out_edges_w_prop(&self, from_vertex: i64) -> impl Iterator<Item = (i64, i64, &E)> {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(move |(_vprop, _ins, outs)| {
                outs.iter()
                    .map(move |(nid2, eprop)| (from_vertex, *nid2, eprop))
            })
    }

    pub fn edges_iter_w_prop(&self) -> impl Iterator<Item = (i64, i64, &E)> {
        self.edges.iter().flat_map(|(nid1, (_vprop, _ins, outs))| {
            outs.iter().map(|(nid2, eprop)| (*nid1, *nid2, eprop))
        })
    }
    pub fn edges_iter_w_prop_mut(&mut self) -> impl Iterator<Item = (i64, i64, &mut E)> {
        self.edges
            .iter_mut()
            .flat_map(|(nid1, (_vprop, _ins, outs))| {
                outs.iter_mut().map(|(nid2, eprop)| (*nid1, *nid2, eprop))
            })
    }

    pub fn edges_par_iter_w_prop_mut(
        &mut self,
    ) -> impl ParallelIterator<Item = (i64, i64, &mut E)> {
        self.edges
            .par_iter_mut()
            .flat_map(|(nid1, (_vprop, _ins, outs))| {
                outs.par_iter_mut()
                    .map(|(nid2, eprop)| (*nid1, *nid2, eprop))
            })
    }

    pub fn assert_consistancy(&self) {
        for (nid1, (_vprop, ins, outs)) in self.edges.iter() {
            // no self loops
            assert!(!ins.contains(nid1));
            assert!(
                !outs.iter().any(|(nid2, _)| nid1 == nid2),
                "{:?} {:?}",
                nid1,
                outs
            );

            // if there's an in, there's an out
            for in_v in ins.iter() {
                assert!(
                    self.edges.contains_key(in_v),
                    "Node {nid1} has an in edge from {in_v}, but there is no data for that vertex {in_v}"
                );
                assert!(
                    self.edges
                        .get(in_v)
                        .unwrap()
                        .2
                        .iter()
                        .any(|(otherv, _eprop)| otherv == nid1),
                    "Graph Data inconsistancy. {nid1} has {in_v} as one of it's in edges, but there is no corresponding out edge from {in_v} to {nid1}"
                );
            }
            // if there's an out, there's an in
            for (out_v, _eprop) in outs.iter() {
                assert!(
                    self.edges.contains_key(out_v),
                    "Node {nid1} has an out edge to {out_v}, but there is no data for that vertex {out_v}"
                );
                assert!(self.edges.get(out_v).unwrap().1.contains(nid1));
            }
        }

        assert_eq!(
            self.edges
                .par_iter()
                .map(|(_nid2, (_vprop, ins, _outs))| ins.len())
                .sum::<usize>(),
            self.edges
                .par_iter()
                .map(|(_nid2, (_vprop, _ins, outs))| outs.len())
                .sum::<usize>()
        );
    }

    /// Remove this vertex, returning the
    /// Any & all edges connected to this vertex are deleted.
    pub fn remove_vertex(&mut self, vertex: &i64) -> Option<V> {
        let (vprop, ins, outs) = self.edges.remove(vertex)?;
        for in_v in ins.iter() {
            self.delete_edge(in_v, vertex);
        }
        for (out_v, _eprop) in outs.iter() {
            self.delete_edge(vertex, out_v);
        }
        Some(vprop)
    }

    pub fn remove_edge(&mut self, vertex1: &i64, vertex2: &i64) -> Option<E> {
        let outs = &mut self.edges.get_mut(vertex1)?.2;
        let idx = outs.iter().position(|(v, _eprop)| v == vertex2)?;
        let (_v2, eprop) = outs.remove(idx);

        // remove the corresponding in edge
        if let Some(x) = self.edges.get_mut(vertex2) {
            x.1.retain(|v1| v1 != vertex1);
        }

        self.delete_vertex_if_unconnected(vertex1);
        self.delete_vertex_if_unconnected(vertex2);

        Some(eprop)
    }

    pub fn vertexes_w_prop(&self) -> impl Iterator<Item = (i64, &V)> + '_ {
        self.edges
            .iter()
            .map(|(v, (vprop, _ins, _outs))| (*v, vprop))
    }

    pub fn vertexes_w_prop_par_mut(&mut self) -> impl ParallelIterator<Item = (i64, &mut V)> {
        self.edges
            .par_iter_mut()
            .map(|(v, (vprop, _ins, _outs))| (*v, vprop))
    }

    pub fn edges_w_prop_par_mut(&mut self) -> impl ParallelIterator<Item = ((i64, i64), &mut E)> {
        self.edges
            .par_iter_mut()
            .flat_map(|(v1, (_vprop, _ins, outs))| {
                outs.par_iter_mut().map(|(v2, eprop)| ((*v1, *v2), eprop))
            })
    }
}

impl<V, E> DirectedGraphTrait for DirectedGraph2<V, E>
where
    V: Send + Default + Clone + Sync + Debug,
    E: Send + Default + Clone + Sync + Debug,
{
    fn new() -> Self {
        Default::default()
    }

    fn out_neighbours(&self, from_vertex: i64) -> impl Iterator<Item = i64> {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(|(_vprop, _ins, outs)| outs.iter().map(|(vid, _eprop)| vid))
            .copied()
    }
    fn in_neighbours(&self, from_vertex: i64) -> impl Iterator<Item = i64> {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(|(_vprop, ins, _outs)| ins.iter())
            .copied()
    }

    fn num_vertexes(&self) -> usize {
        self.edges.len()
    }
    fn num_edges(&self) -> usize {
        self.edges
            .iter()
            .map(|(_vertexid, (_vprop, _in_edges, out_edges))| out_edges.len())
            .sum()
    }
    /// True iff this vertex is in this graph
    fn contains_vertex(&self, vid: &i64) -> bool {
        self.edges.contains_key(vid)
    }

    /// Iterator over all edges
    fn edges_iter(&self) -> impl Iterator<Item = (i64, i64)> {
        self.edges
            .iter()
            .flat_map(move |(v, (_vprop, _ins, outs))| outs.iter().map(move |(o, _eprop)| (*v, *o)))
    }
    fn edges_par_iter(&self) -> impl ParallelIterator<Item = (i64, i64)> {
        self.edges_iter().par_bridge()
    }

    /// returns each vertex and the number of out edges
    fn vertexes_and_num_outs(&self) -> impl Iterator<Item = (i64, usize)> {
        self.edges
            .iter()
            .map(|(v, (_vprop, _ins, outs))| (*v, outs.len()))
    }

    fn vertex_has_outgoing(&self, vid: &i64) -> bool {
        self.edges
            .get(vid)
            .is_some_and(|(_vprop, _ins, outs)| !outs.is_empty())
    }

    fn detailed_size(&self) -> String {
        let s = format!(
            "DirectedGraph2: num_vertexes {} num_edges {}",
            self.num_vertexes(),
            self.num_edges(),
        );
        //s.push_str(&format!(
        //    "\nSize of graph: {} = {} bytes.\nbytes/vertex = {:>.5}\nbytes/edge = {:>.5}",
        //    self.get_size(),
        //    self.get_size().to_formatted_string(&Locale::en),
        //    self.get_size() as f64 / self.num_vertexes() as f64,
        //    self.get_size() as f64 / self.num_edges() as f64,
        //));

        s
    }

    fn num_ins_zero(&self, vid: &i64) -> bool {
        self.edges
            .get(vid)
            .is_none_or(|(_vprop, ins, _outs)| ins.is_empty())
    }

    fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    fn vertexes_and_num_ins_outs(&self) -> impl Iterator<Item = (i64, usize, usize)> + '_ {
        self.edges
            .iter()
            .map(|(v, (_vprop, ins, outs))| (*v, ins.len(), outs.len()))
    }

    fn neighbors_in_xor_out(&mut self, v: &i64) -> bool {
        self.edges
            .get(v)
            .is_some_and(|(_vprop, ins, outs)| !ins.is_empty() ^ !outs.is_empty())
    }
    fn delete_edge(&mut self, vertex1: &i64, vertex2: &i64) {
        self.remove_edge(vertex1, vertex2);
    }

    fn delete_vertex(&mut self, vertex: &i64) {
        self.remove_vertex(vertex);
    }

    fn contract_vertex(&mut self, vertex: &i64, replacement: &i64) {
        if vertex == replacement {
            warn!("Trying to contract a vertex with itself: {}", vertex);
            return;
        }
        if !self.contains_vertex(vertex) && !self.contains_vertex(replacement) {
            warn!(
                "Trying to replace the vertex {vertex} with {replacement}, but neither are in the graph"
            );
            return;
        }
        assert!(
            self.contains_vertex(vertex),
            "Vertex to replace {vertex} doesn't exist"
        );
        assert!(
            self.contains_vertex(replacement),
            "Replacement vertex {replacement} doesn't exist"
        );

        let mut old = match self.edges.remove(vertex) {
            None => {
                return;
            }
            Some(old) => old,
        };

        self.set_vertex_property(replacement, old.0);

        for (out_v, eprop) in old.2.drain(..) {
            self.edges
                .get_mut(&out_v)
                .unwrap()
                .1
                .retain(|in_v| in_v != vertex);
            self.add_edge_w_prop(*replacement, out_v, eprop);
        }
        for in_v in old.1.iter() {
            if let Some(eprop) = self.remove_edge(in_v, vertex) {
                self.add_edge_w_prop(*in_v, *replacement, eprop);
            }
        }
    }

    fn vertexes(&self) -> impl Iterator<Item = i64> + '_ {
        self.edges.keys().copied()
    }

    /// Adds an edge between these 2, returning true iff the edge already existed
    fn add_edge(&mut self, vertex1: i64, vertex2: i64) -> bool {
        if vertex1 == vertex2 {
            return false;
        }
        let from_v1 = &mut self.edges.entry(vertex1).or_default().2;
        if from_v1.iter().any(|(vid, _eprop)| vid == &vertex2) {
            return true;
        } else {
            from_v1.push((vertex2, E::default()));
        }

        // assume we never get inconsistant
        let other = self.edges.entry(vertex2).or_default();
        other.1.push(vertex1);
        other.1.sort();
        false
    }

    fn into_disconnected_graphs(self, progress_bar: &ProgressBar) -> impl Iterator<Item = Self> {
        dbg!("here");
        let mut g = self;
        let mut vertexes_to_look_at = Vec::new();

        std::iter::from_fn(move || {
            if g.is_empty() {
                return None;
            }
            let mut new_graph: DirectedGraph2<V, E> = DirectedGraph2::new();
            vertexes_to_look_at.truncate(0);
            vertexes_to_look_at.push(g.vertexes().next().unwrap());
            let mut num_vertexes = 0;

            while let Some(vertex) = vertexes_to_look_at.pop() {
                num_vertexes += 1;
                if !g.contains_vertex(&vertex) {
                    continue;
                }

                let (vprop, ins, outs) = g.edges.remove(&vertex).unwrap();
                new_graph.add_vertex_w_prop(vertex, vprop);

                for (out_v, eprop) in outs.into_iter() {
                    new_graph.add_edge_w_prop(vertex, out_v, eprop);
                    vertexes_to_look_at.push(out_v);
                }
                vertexes_to_look_at.extend(ins.into_iter());
            }

            progress_bar.inc(num_vertexes as u64);
            Some(new_graph)
        })
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

/// slice::windows, but only does length 2, and it starts at the end of the slice and walks
/// backwards. the tuple of elements is in the original order (see tests above)
pub fn rwindows2<T>(slice: &[T]) -> impl Iterator<Item = (&T, &T)> {
    (1..slice.len()).rev().map(|i| (&slice[i - 1], &slice[i]))
}
