use super::*;
use log::warn;
use rayon::prelude::ParallelIterator;
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashSet};
use std::fmt::Debug;

use crate::kosaraju;
use itertools::Itertools;
use std::iter;

pub trait DirectedGraphTrait<V, E>: Send + Sync + Sized {
    fn new() -> Self;

    fn in_neighbours(&self, vertex: i64) -> impl Iterator<Item = i64>;
    fn num_in_neighbours(&self, vertex: i64) -> Option<usize> {
        if self.contains_vertex(&vertex) {
            Some(self.in_neighbours(vertex).count())
        } else {
            None
        }
    }
    /// Returns (a, to_vertex) where a is an in neighbour of to_vertex
    fn in_edges(&self, vertex: i64) -> impl Iterator<Item = (i64, i64)> {
        self.in_neighbours(vertex).map(move |nid0| (nid0, vertex))
    }
    fn out_neighbours(&self, vertex: i64) -> impl Iterator<Item = i64>;
    fn out_neighbours_w_prop<'a>(&'a self, from_vertex: i64) -> impl Iterator<Item = (i64, &'a E)>
    where
        E: 'a;
    fn num_out_neighbours(&self, vertex: i64) -> Option<usize> {
        if self.contains_vertex(&vertex) {
            Some(self.out_neighbours(vertex).count())
        } else {
            None
        }
    }

    /// Returns (from_vertex, b) where b is an out neighbour of from_vertex
    fn out_edges(&self, vertex: i64) -> impl Iterator<Item = (i64, i64)> {
        self.out_neighbours(vertex).map(move |nid2| (vertex, nid2))
    }

    /// All edges that go to/from this vertex. No guarantee of order.
    fn edges(&self, vertex: i64) -> impl Iterator<Item = (i64, i64)> {
        self.in_edges(vertex).chain(self.out_edges(vertex))
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
        self.num_in_neighbours(*vid) == Some(0)
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
        (self.num_in_neighbours(*v) == Some(0)) ^ (self.num_out_neighbours(*v) == Some(0))
    }
    fn delete_edge(&mut self, vertex1: &i64, vertex2: &i64);

    fn expand_edge(&self, vertex1: i64, vertex2: i64) -> impl Iterator<Item = i64> + '_ {
        iter::once(vertex1).chain(iter::once(vertex2))
    }

    /// Removes this vertex (& associated edges) from this graph
    fn delete_vertex(&mut self, vertex: &i64);

    fn contract_vertex(&mut self, vertex: &i64, replacement: &i64);
    /// Iterator over all vertexes
    fn vertexes(&self) -> impl Iterator<Item = i64> + '_ {
        self.vertexes_iter()
    }
    fn vertexes_iter(&self) -> impl Iterator<Item = i64> + '_;
    fn vertexes_par_iter(&self) -> impl ParallelIterator<Item = i64>;

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
        if self.num_in_neighbours(*vertex) == Some(0) && self.num_out_neighbours(*vertex) == Some(0)
        {
            self.delete_vertex(vertex);
        }
    }

    fn vertex_property(&self, vertex: &i64) -> Option<&V>;
    fn vertex_property_unchecked(&self, vertex: &i64) -> &V;

    fn set_vertex_property(&mut self, vertex: &i64, property: V);
    fn vertex_property_mut(&mut self, vertex: &i64) -> &mut V;

    fn edge_property(&self, edge: (i64, i64)) -> Option<&E>;
    fn edge_property_unchecked(&self, edge: (i64, i64)) -> &E;

    fn set_edge_property(&mut self, edge: (i64, i64), property: E);

    fn edge_property_mut(&mut self, edge: (i64, i64)) -> &mut E;

    fn add_edge_w_prop(&mut self, vertex1: i64, vertex2: i64, eprop: E);

    fn add_vertex_w_prop(&mut self, vertex: i64, vprop: V);

    /// Returns (from_vertex, b, E) where b is an out neighbour of from_vertex
    fn out_edges_w_prop<'a>(&'a self, from_vertex: i64) -> impl Iterator<Item = (i64, i64, &'a E)>
    where
        E: 'a;
    fn out_edges_w_prop_mut<'a>(
        &'a mut self,
        from_vertex: i64,
    ) -> impl Iterator<Item = (i64, i64, &'a mut E)>
    where
        E: 'a;
    fn in_edges_w_prop<'a>(&'a self, to_vertex: i64) -> impl Iterator<Item = (i64, i64, &'a E)>
    where
        E: 'a;

    fn edges_iter_w_prop<'a>(&'a self) -> impl Iterator<Item = (i64, i64, &'a E)>
    where
        E: 'a;
    fn edges_iter_w_prop_mut<'a>(&'a mut self) -> impl Iterator<Item = (i64, i64, &'a mut E)>
    where
        E: 'a;
    fn edges_par_iter_w_prop<'a>(&'a self) -> impl ParallelIterator<Item = (i64, i64, &'a E)>
    where
        E: 'a;

    fn edges_par_iter_w_prop_mut<'a>(
        &'a mut self,
    ) -> impl ParallelIterator<Item = (i64, i64, &'a mut E)>
    where
        E: 'a;

    fn assert_consistancy(&self);

    /// Remove this vertex, returning the
    /// Any & all edges connected to this vertex are deleted.
    fn remove_vertex(&mut self, vertex: &i64) -> Option<V>;

    fn remove_edge(&mut self, vertex1: &i64, vertex2: &i64) -> Option<E>;

    fn vertexes_w_prop<'a>(&'a self) -> impl Iterator<Item = (i64, &'a V)>
    where
        V: 'a;
    fn vertexes_w_prop_par<'a>(&'a self) -> impl ParallelIterator<Item = (i64, &'a V)>
    where
        V: 'a;
    fn vertexes_w_prop_par_mut<'a>(&'a mut self) -> impl ParallelIterator<Item = (i64, &'a mut V)>
    where
        V: 'a;

    fn edges_w_prop_par_mut<'a>(
        &'a mut self,
    ) -> impl ParallelIterator<Item = ((i64, i64), &'a mut E)>
    where
        E: 'a;
}

#[derive(Default, Debug, Clone)]
struct Vertex<V, E> {
    vprop: V,
    ins: SmallVec<[i64; 1]>,
    outs: SmallVec<[(i64, E); 1]>,
}

impl<V, E> Vertex<V, E> {
    #[allow(clippy::type_complexity)]
    fn into_parts(self) -> (V, SmallVec<[i64; 1]>, SmallVec<[(i64, E); 1]>) {
        (self.vprop, self.ins, self.outs)
    }
}

/// A graph which stores a list of all incoming and outgoing edges
#[derive(Default, Debug, Clone)]
pub struct DirectedGraph<V, E>
where
    V: Send + Default + Clone + Sync + Debug,
    E: Send + Default + Clone + Sync + Debug,
{
    // key is vertex id
    edges: BTreeMap<i64, Vertex<V, E>>,
}

impl<V, E> DirectedGraph<V, E>
where
    V: Send + Default + Clone + Sync + Debug,
    E: Send + Default + Clone + Sync + Debug,
{
}

impl<V, E> DirectedGraphTrait<V, E> for DirectedGraph<V, E>
where
    V: Send + Default + Clone + Sync + Debug,
    E: Send + Default + Clone + Sync + Debug,
{
    fn new() -> Self {
        Default::default()
    }
    fn num_in_neighbours(&self, vertex: i64) -> Option<usize> {
        self.edges.get(&vertex).map(|v| v.ins.len())
    }
    fn num_out_neighbours(&self, vertex: i64) -> Option<usize> {
        self.edges.get(&vertex).map(|v| v.outs.len())
    }

    fn out_neighbours(&self, from_vertex: i64) -> impl Iterator<Item = i64> {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(|v| v.outs.iter().map(|(nid, _eprop)| nid))
            .copied()
    }
    fn out_neighbours_w_prop<'a>(&'a self, from_vertex: i64) -> impl Iterator<Item = (i64, &'a E)>
    where
        E: 'a,
    {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(|v| v.outs.iter())
            .map(|(nid2, eprop)| (*nid2, eprop))
    }
    fn in_neighbours(&self, from_vertex: i64) -> impl Iterator<Item = i64> {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(|v| v.ins.iter())
            .copied()
    }

    fn num_vertexes(&self) -> usize {
        self.edges.len()
    }
    fn num_edges(&self) -> usize {
        self.edges.values().map(|v| v.outs.len()).sum()
    }
    /// True iff this vertex is in this graph
    fn contains_vertex(&self, vid: &i64) -> bool {
        self.edges.contains_key(vid)
    }

    /// Iterator over all edges
    fn edges_iter(&self) -> impl Iterator<Item = (i64, i64)> {
        self.edges
            .iter()
            .flat_map(move |(nid, v)| v.outs.iter().map(move |(o, _eprop)| (*nid, *o)))
    }
    fn edges_par_iter(&self) -> impl ParallelIterator<Item = (i64, i64)> {
        self.edges_iter().par_bridge()
    }

    /// returns each vertex and the number of out edges
    fn vertexes_and_num_outs(&self) -> impl Iterator<Item = (i64, usize)> {
        self.edges.iter().map(|(nid1, v)| (*nid1, v.outs.len()))
    }

    fn vertex_has_outgoing(&self, vid: &i64) -> bool {
        self.edges.get(vid).is_some_and(|v| !v.outs.is_empty())
    }

    fn detailed_size(&self) -> String {
        let s = format!(
            "DirectedGraph: num_vertexes {} num_edges {}",
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
        self.edges.get(vid).is_none_or(|v| v.ins.is_empty())
    }

    fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    fn vertexes_and_num_ins_outs(&self) -> impl Iterator<Item = (i64, usize, usize)> + '_ {
        self.edges
            .iter()
            .map(|(nid, v)| (*nid, v.ins.len(), v.outs.len()))
    }

    fn neighbors_in_xor_out(&mut self, v: &i64) -> bool {
        self.edges
            .get(v)
            .is_some_and(|v| !v.ins.is_empty() ^ !v.outs.is_empty())
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

        self.set_vertex_property(replacement, old.vprop);

        for (out_v, eprop) in old.outs.drain(..) {
            self.edges
                .get_mut(&out_v)
                .unwrap()
                .ins
                .retain(|in_v| in_v != vertex);
            self.add_edge_w_prop(*replacement, out_v, eprop);
        }
        for in_v in old.ins.iter() {
            if let Some(eprop) = self.remove_edge(in_v, vertex) {
                self.add_edge_w_prop(*in_v, *replacement, eprop);
            }
        }
    }

    fn vertexes_iter(&self) -> impl Iterator<Item = i64> + '_ {
        self.edges.keys().copied()
    }
    fn vertexes_par_iter(&self) -> impl ParallelIterator<Item = i64> {
        self.edges.par_iter().map(|(nid, _)| nid).copied()
    }

    /// Adds an edge between these 2, returning true iff the edge already existed
    fn add_edge(&mut self, vertex1: i64, vertex2: i64) -> bool {
        if vertex1 == vertex2 {
            return false;
        }
        let from_v1 = &mut self.edges.entry(vertex1).or_default().outs;
        if from_v1.iter().any(|(vid, _eprop)| vid == &vertex2) {
            return true;
        } else {
            from_v1.push((vertex2, E::default()));
        }

        // assume we never get inconsistant
        let other = self.edges.entry(vertex2).or_default();
        other.ins.push(vertex1);
        other.ins.sort();
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
            let mut new_graph: DirectedGraph<V, E> = DirectedGraph::new();
            vertexes_to_look_at.truncate(0);
            vertexes_to_look_at.push(g.vertexes().next().unwrap());
            let mut num_vertexes = 0;

            while let Some(vertex) = vertexes_to_look_at.pop() {
                num_vertexes += 1;
                if !g.contains_vertex(&vertex) {
                    continue;
                }

                let (vprop, ins, outs) = g.edges.remove(&vertex).unwrap().into_parts();
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

    fn vertex_property(&self, vertex: &i64) -> Option<&V> {
        self.edges.get(vertex).map(|v| &v.vprop)
    }
    fn vertex_property_unchecked(&self, vertex: &i64) -> &V {
        self.vertex_property(vertex).unwrap()
    }

    fn set_vertex_property(&mut self, vertex: &i64, property: V) {
        self.edges.entry(*vertex).or_default().vprop = property;
    }
    fn vertex_property_mut(&mut self, vertex: &i64) -> &mut V {
        &mut self.edges.entry(*vertex).or_default().vprop
    }

    fn edge_property(&self, edge: (i64, i64)) -> Option<&E> {
        self.edges.get(&edge.0).map(|v| &v.outs).and_then(|outs| {
            outs.iter()
                .find(|(vid, _p)| *vid == edge.1)
                .map(|(_vid, prop)| prop)
        })
    }
    fn edge_property_unchecked(&self, edge: (i64, i64)) -> &E {
        self.edge_property(edge).unwrap()
    }

    fn set_edge_property(&mut self, edge: (i64, i64), property: E) {
        if edge.0 == edge.1 {
            return;
        }
        *self.edge_property_mut(edge) = property;
    }

    fn edge_property_mut(&mut self, edge: (i64, i64)) -> &mut E {
        let vertex = self.edges.get_mut(&edge.0).unwrap();
        let outs = &mut vertex.outs;
        if !outs.iter().any(|(oid, _eprop)| *oid == edge.1) {
            outs.push((edge.1, E::default()));
        }

        outs.iter_mut()
            .filter(|(oid, _)| *oid == edge.1)
            .map(|(_, eprop)| eprop)
            .next()
            .unwrap()
    }

    fn add_edge_w_prop(&mut self, vertex1: i64, vertex2: i64, eprop: E) {
        if vertex1 == vertex2 {
            return;
        }
        self.add_edge(vertex1, vertex2);
        self.set_edge_property((vertex1, vertex2), eprop);
    }

    fn add_vertex_w_prop(&mut self, vertex: i64, vprop: V) {
        // TODO remove all clones
        self.edges
            .entry(vertex)
            .and_modify(|o| o.vprop = vprop.clone())
            .or_insert(Vertex {
                vprop,
                ..Default::default()
            });
    }

    /// Returns (from_vertex, b, E) where b is an out neighbour of from_vertex
    fn out_edges_w_prop<'a>(&'a self, from_vertex: i64) -> impl Iterator<Item = (i64, i64, &'a E)>
    where
        E: 'a,
    {
        self.edges.get(&from_vertex).into_iter().flat_map(move |v| {
            v.outs
                .iter()
                .map(move |(nid2, eprop)| (from_vertex, *nid2, eprop))
        })
    }
    /// Returns (from_vertex, b, E) where b is an out neighbour of from_vertex
    fn out_edges_w_prop_mut<'a>(
        &'a mut self,
        from_vertex: i64,
    ) -> impl Iterator<Item = (i64, i64, &'a mut E)>
    where
        E: 'a,
    {
        self.edges
            .get_mut(&from_vertex)
            .into_iter()
            .flat_map(move |v| {
                v.outs
                    .iter_mut()
                    .map(move |(nid2, eprop)| (from_vertex, *nid2, eprop))
            })
    }
    fn in_edges_w_prop<'a>(&'a self, to_vertex: i64) -> impl Iterator<Item = (i64, i64, &'a E)>
    where
        E: 'a,
    {
        self.in_edges(to_vertex)
            .map(|(nid1, nid2)| (nid1, nid2, self.edge_property_unchecked((nid1, nid2))))
    }

    fn edges_iter_w_prop<'a>(&'a self) -> impl Iterator<Item = (i64, i64, &'a E)>
    where
        E: 'a,
    {
        self.edges
            .iter()
            .flat_map(|(nid1, v)| v.outs.iter().map(|(nid2, eprop)| (*nid1, *nid2, eprop)))
    }
    fn edges_iter_w_prop_mut<'a>(&'a mut self) -> impl Iterator<Item = (i64, i64, &'a mut E)>
    where
        E: 'a,
    {
        self.edges
            .iter_mut()
            .flat_map(|(nid1, v)| v.outs.iter_mut().map(|(nid2, eprop)| (*nid1, *nid2, eprop)))
    }
    fn edges_par_iter_w_prop<'a>(&'a self) -> impl ParallelIterator<Item = (i64, i64, &'a E)>
    where
        E: 'a,
    {
        self.edges
            .par_iter()
            .flat_map(|(nid1, v)| v.outs.par_iter().map(|(nid2, eprop)| (*nid1, *nid2, eprop)))
    }

    fn edges_par_iter_w_prop_mut<'a>(
        &'a mut self,
    ) -> impl ParallelIterator<Item = (i64, i64, &'a mut E)>
    where
        E: 'a,
    {
        self.edges.par_iter_mut().flat_map(|(nid1, v)| {
            v.outs
                .par_iter_mut()
                .map(|(nid2, eprop)| (*nid1, *nid2, eprop))
        })
    }

    fn assert_consistancy(&self) {
        for (nid1, v) in self.edges.iter() {
            // no self loops
            assert!(!v.ins.contains(nid1));
            assert!(
                !v.outs.iter().any(|(nid2, _)| nid1 == nid2),
                "{:?} {:?}",
                nid1,
                v.outs
            );

            // if there's an in, there's an out
            for in_v in v.ins.iter() {
                assert!(
                    self.edges.contains_key(in_v),
                    "Node {nid1} has an in edge from {in_v}, but there is no data for that vertex {in_v}"
                );
                assert!(
                    self.edges
                        .get(in_v)
                        .unwrap()
                        .outs
                        .iter()
                        .any(|(otherv, _eprop)| otherv == nid1),
                    "Graph Data inconsistancy. {nid1} has {in_v} as one of it's in edges, but there is no corresponding out edge from {in_v} to {nid1}"
                );
            }
            // if there's an out, there's an in
            for (out_v, _eprop) in v.outs.iter() {
                assert!(
                    self.edges.contains_key(out_v),
                    "Node {nid1} has an out edge to {out_v}, but there is no data for that vertex {out_v}"
                );
                assert!(self.edges.get(out_v).unwrap().ins.contains(nid1));
            }
        }

        assert_eq!(
            self.edges
                .par_iter()
                .map(|(_nid2, v)| v.ins.len())
                .sum::<usize>(),
            self.edges
                .par_iter()
                .map(|(_nid2, v)| v.outs.len())
                .sum::<usize>()
        );
    }

    /// Remove this vertex, returning the
    /// Any & all edges connected to this vertex are deleted.
    fn remove_vertex(&mut self, vertex: &i64) -> Option<V> {
        let (vprop, ins, outs) = self.edges.remove(vertex)?.into_parts();
        for in_v in ins.iter() {
            self.delete_edge(in_v, vertex);
        }
        for (out_v, _eprop) in outs.iter() {
            self.delete_edge(vertex, out_v);
        }
        Some(vprop)
    }

    fn remove_edge(&mut self, vertex1: &i64, vertex2: &i64) -> Option<E> {
        let outs = &mut self.edges.get_mut(vertex1)?.outs;
        let idx = outs.iter().position(|(v, _eprop)| v == vertex2)?;
        let (_v2, eprop) = outs.remove(idx);

        // remove the corresponding in edge
        if let Some(x) = self.edges.get_mut(vertex2) {
            x.ins.retain(|v1| v1 != vertex1);
        }

        self.delete_vertex_if_unconnected(vertex1);
        self.delete_vertex_if_unconnected(vertex2);

        Some(eprop)
    }

    fn vertexes_w_prop<'a>(&'a self) -> impl Iterator<Item = (i64, &'a V)>
    where
        V: 'a,
    {
        self.edges.iter().map(|(nid, v)| (*nid, &v.vprop))
    }

    fn vertexes_w_prop_par_mut<'a>(&'a mut self) -> impl ParallelIterator<Item = (i64, &'a mut V)>
    where
        V: 'a,
    {
        self.edges
            .par_iter_mut()
            .map(|(nid, v)| (*nid, &mut v.vprop))
    }

    fn vertexes_w_prop_par<'a>(&'a self) -> impl ParallelIterator<Item = (i64, &'a V)>
    where
        V: 'a,
    {
        self.edges.par_iter().map(|(nid, v)| (*nid, &v.vprop))
    }

    fn edges_w_prop_par_mut<'a>(
        &'a mut self,
    ) -> impl ParallelIterator<Item = ((i64, i64), &'a mut E)>
    where
        E: 'a,
    {
        self.edges.par_iter_mut().flat_map(|(nid1, v)| {
            v.outs
                .par_iter_mut()
                .map(|(nid2, eprop)| ((*nid1, *nid2), eprop))
        })
    }
}
