#![allow(dead_code)]
use super::*;
use anyhow::{Context, Result};
use rayon::prelude::ParallelIterator;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::iter::once;

use crate::btreemapsplitkey::BTreeMapSplitKey;
use crate::kosaraju;

pub(crate) struct UndirectedGraph<T>
where
    T: Clone,
{
    data: Vec<T>,
    size: usize,
}

// This code could probably be deleted. It's not used.
#[allow(unused)]
impl<T> UndirectedGraph<T>
where
    T: Clone,
{
    pub fn new(size: usize, initial: T) -> Result<Self> {
        // FIXME this needs too much memory if size is large
        //dbg!("Before graph alloc size", size);
        let mut data = Vec::new();
        data.try_reserve(size * size).with_context(|| {
            format!(
                "Tried to allocate {size}*{size} (total {total}) bytes for graph",
                size = size,
                total = (size * size)
            )
        })?;
        for _ in 0..(size * size) {
            data.push(initial.clone());
        }
        //dbg!("after graph alloc size", size);
        Ok(UndirectedGraph { data, size })
    }

    pub fn len(&self) -> usize {
        self.size
    }

    #[allow(unused)]
    pub fn get(&self, i: usize, j: usize) -> &T {
        if i < j {
            &self.data[i * self.size + j]
        } else {
            &self.data[j * self.size + i]
        }
    }
    #[allow(unused)]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        if i < j {
            &mut self.data[i * self.size + j]
        } else {
            &mut self.data[j * self.size + i]
        }
    }
    #[allow(unused)]
    pub fn set_single(&mut self, i: usize, j: usize, val: T) {
        self.data[i * self.size + j] = val.clone();
    }
    pub fn set(&mut self, i: usize, j: usize, val: T) {
        self.data[i * self.size + j] = val.clone();
        self.data[j * self.size + i] = val;
    }

    pub fn values(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        (0..self.size)
            .flat_map(|i| (0..self.size).map(move |j| (i, j)))
            .map(|(i, j)| (i, j, &self.data[i * self.size + j]))
    }

    pub fn neighbors(&self, nid: usize) -> impl Iterator<Item = (usize, &T)> {
        (0..self.size)
            .filter(move |i| *i != nid)
            .map(move |i| (i, self.get(nid, i)))
    }

    pub fn pretty_print(&self, fmt: &dyn Fn(&T) -> String, col_join: &str) -> String {
        let mut strs: Vec<Vec<String>> = (0..self.size)
            .map(|i| {
                (0..self.size)
                    .map(|j| fmt(&self.data[i * self.size + j]))
                    .collect()
            })
            .collect();
        let max = strs
            .iter()
            .flat_map(|row| row.iter())
            .map(|el| el.len())
            .max()
            .unwrap_or(1);

        strs.par_iter_mut().for_each(|row| {
            row.par_iter_mut().for_each(|val| {
                *val = format!("{0:>1$}", val, max);
            })
        });

        strs.into_iter()
            .map(|row| row.join(col_join))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

pub(crate) struct DirectedGraph<T>
where
    T: Clone,
{
    data: Vec<T>,
    size: usize,
}

#[allow(unused)]
impl<T> DirectedGraph<T>
where
    T: Clone,
{
    pub fn new(size: usize, initial: T) -> Self {
        DirectedGraph {
            data: vec![initial; size * size],
            size,
        }
    }

    #[allow(unused)]
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn get(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.size + j]
    }
    #[allow(unused)]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.size + j]
    }
    pub fn set(&mut self, i: usize, j: usize, val: T) {
        assert!(i <= self.size);
        assert!(j <= self.size);
        self.data[i * self.size + j] = val;
    }

    pub fn values(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        (0..self.size)
            .flat_map(|i| (0..self.size).map(move |j| (i, j)))
            .map(|(i, j)| (i, j, &self.data[i * self.size + j]))
    }

    pub fn pretty_print(&self, fmt: &dyn Fn(&T) -> String, col_join: &str) -> String {
        let mut strs: Vec<Vec<String>> = (0..self.size)
            .map(|i| {
                (0..self.size)
                    .map(|j| fmt(&self.data[i * self.size + j]))
                    .collect()
            })
            .collect();
        let max = strs
            .iter()
            .flat_map(|row| row.iter())
            .map(|el| el.len())
            .max()
            .unwrap_or(1);

        strs.par_iter_mut().for_each(|row| {
            row.par_iter_mut().for_each(|val| {
                *val = format!("{0:>1$}", val, max);
            })
        });

        strs.into_iter()
            .map(|row| row.join(col_join))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

type SmallVecIntermediates<V> = SmallVec<[V; 1]>;

pub(crate) struct UndirectedAdjGraph<V, E> {
    edges: BTreeMap<V, BTreeMap<V, (E, SmallVecIntermediates<V>)>>,
}

#[allow(dead_code)]
impl<V, E> UndirectedAdjGraph<V, E>
where
    V: std::hash::Hash + Eq + Copy + Ord + Send + std::fmt::Debug + Default,
    E: Copy + PartialOrd + Clone + std::fmt::Debug + std::ops::Add<Output = E> + std::cmp::PartialEq + Default,
{
    pub fn new() -> Self {
        Self {
            edges: Default::default()
        }
    }

    pub fn set(&mut self, i: &V, j: &V, val: E) {
        self.edges.entry(*i).or_default().insert(*j, (val, Default::default()));
        self.edges.entry(*j).or_default().insert(*i, (val, Default::default()));
    }

    pub fn remove_vertex(&mut self, v: &V) {
        while let Some((other_v, (_weight, _intermediaters))) = self.edges.get_mut(v).unwrap().pop_last() {
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

    pub fn get_all(&self, i: &V, j: &V) -> Option<&(E, SmallVecIntermediates<V>)> {
        self.edges.get(i).and_then(|from_i| from_i.get(j))
    }

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
                (edge_weight, once(a).chain(inter.iter()).chain(once(b)))
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
    pub fn neighbors(&self, i: &V) -> impl Iterator<Item = (&V, &E)> {
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
    pub fn pretty_print(&self, _fmt: &dyn Fn(&E) -> String, _col_join: &str) -> String {
        String::new()
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

    /// Contract this vertex, returnign true iff this graph was modified
    pub fn contract_vertex(&mut self, v: &V) -> bool {
        if !self.contains_vertex(v) {
            warn!("Called contract_vertex on v: {:?} which doesn't exist", v);
            return false;
        }
        if self.num_neighbors(v) != 2 {
            trace!(
                "Called contract_vertex on v: {:?} and it has {} ≠ 2 neighbours",
                v,
                self.num_neighbors(v)
            );
            return false;
        }
        // a - v - b
        let a = *self.edges[v].keys().nth(0).unwrap();
        let b = *self.edges[v].keys().nth(1).unwrap();
        assert!(a != b);
        if let Some(weight_a_b) = self.get(&a, &b) {
            // there already is an edge from a↔b
            if *weight_a_b <= *self.get(&a, v).unwrap() + *self.get(v, &b).unwrap() {
                // this route via v is longer, so delete v & don't create any new edges
                self.remove_vertex(v);
                return true;
            } else {
                // this route via v is shorter, so remove the a-b route and let the rest of th
                // TODO rather than delete and go onwards, just update the a-b vertex instead
                // (which should save allocations etc)
                self.remove_edge(&a, &b);
            }
        }
        assert!(self.edges[&a].contains_key(v));
        assert!(self.edges[&b].contains_key(v));
        assert!(
            self.edges[&a][v].0 + self.edges[v][&b].0 == self.edges[&b][v].0 + self.edges[&v][&a].0
        );
        let edge_a_v = self.edges.get_mut(&a).unwrap().remove(v).unwrap();
        let _edge_b_v = self.edges.get_mut(&b).unwrap().remove(v).unwrap();
        let _edge_v_a = self.edges.get_mut(v).unwrap().remove(&a).unwrap();
        let mut edge_v_b = self.edges.get_mut(v).unwrap().remove(&b).unwrap();
        assert!(self.edges[v].is_empty());
        self.edges.remove(v);
        let new_weight = edge_a_v.0 + edge_v_b.0;

        // We need 2 new Vecs for the a→b & b→a intermediates. Rather than create new Vec, here we
        // reuse the vecs from a→v & v→b (which we have already `.remove`'ed above). This reduces
        // allocations, and might speed up the code.
        let (_weight_a_v, mut new_a_b_intermediates) = edge_a_v;
        new_a_b_intermediates.reserve(1 + edge_v_b.1.len());
        new_a_b_intermediates.push(*v);
        new_a_b_intermediates.append(&mut edge_v_b.1);

        let (_weight_v_b, mut new_b_a_intermediates) = edge_v_b;
        // New b_a Vec needs to be as big as a_b. Resize with default value of v for now (it'll be
        // overwritten later)
        new_b_a_intermediates.resize(new_a_b_intermediates.len(), *v);
        new_b_a_intermediates.copy_from_slice(&new_a_b_intermediates);
        new_b_a_intermediates.reverse();

        let new_edge_a_b = (new_weight, new_a_b_intermediates);
        let new_edge_b_a = (new_weight, new_b_a_intermediates);

        self.edges.get_mut(&a).unwrap().insert(b, new_edge_a_b);
        self.edges.get_mut(&b).unwrap().insert(a, new_edge_b_a);

        true
    }

    /// Returns true iff the graph was modified
    pub fn contract_edges(&mut self) -> bool {
        self.contract_edges_some(|_| true)
    }

    /// Contract the edges, but only contract (remove) vertexes that return true.
    /// Returns true iff the graph was modified
    pub fn contract_edges_some(&mut self, can_contract_vertex: impl Fn(&V) -> bool) -> bool {
        let mut graph_has_been_modified = false;
        let initial_num_edges = self.num_edges();
        let initial_num_vertexes = self.num_vertexes();
        trace!(
            "Starting contract_edges with {} edges and {} vertexes",
            initial_num_edges,
            initial_num_vertexes
        );
        if initial_num_edges == 1 {
            return false;
        }

        let mut graph_has_been_modified_this_iteration;
        let mut candidate_vertexes = Vec::new();
        let mut contraction_round = 0;
        let mut this_vertex_contracted;
        loop {
            trace!(
                "Contraction round {}. There are {} vertexes and {} edges",
                contraction_round,
                self.num_vertexes(),
                self.num_edges()
            );
            contraction_round += 1;
            candidate_vertexes.extend(
                self.iter_vertexes_num_neighbours()
                    .filter_map(|(v, nn)| if nn == 2 { Some(v) } else { None })
                    .filter(|v| can_contract_vertex(v))
                    .cloned(),
            );
            if candidate_vertexes.is_empty() {
                trace!("No more candidate vertexes");
                break;
            }
            trace!("There are {} candidate vertexes", candidate_vertexes.len());
            graph_has_been_modified_this_iteration = false;
            for v in candidate_vertexes.drain(..) {
                this_vertex_contracted = self.contract_vertex(&v);
                if this_vertex_contracted {
                    //trace!("Vertex {:?} was contracted", v);
                    graph_has_been_modified_this_iteration = true;
                    graph_has_been_modified = true;
                } else {
                    //trace!("Vertex {:?} was not contracted", v);
                }
            }

            if !graph_has_been_modified_this_iteration {
                trace!("End of loop, and no changes made → break out");
                break;
            }
        }

        debug!("End of contract_edges there are now {} edges and {} vertexes. Removed {} edges and {} vertexes in {} rounds", self.num_edges(), self.num_vertexes(), initial_num_edges-self.num_edges(), initial_num_vertexes-self.num_vertexes(), contraction_round);
        graph_has_been_modified
    }


    /// Spike = vertex with just one neighbour.
    /// Returns true iff graph has been modifed
    pub fn remove_spikes(&mut self, can_contract_vertex: impl Fn(&V) -> bool) -> bool {
        let initial_num_edges = self.num_edges();
        //let initial_num_vertexes = self.num_vertexes();
        if initial_num_edges == 1 {
            return false;
        }

        let mut graph_has_been_modified = false;
        let mut graph_has_been_modified_this_iteration;
        let mut candidate_vertexes = Vec::new();
        //let mut contraction_round = 0;
        loop {
            //contraction_round += 1;
            candidate_vertexes.extend(
                self.iter_vertexes_num_neighbours()
                    .filter_map(|(v, nn)| if nn == 1 { Some(v) } else { None })
                    .filter(|v| can_contract_vertex(v))
                    .cloned(),
            );
            if candidate_vertexes.is_empty() {
                trace!("No more candidate vertexes");
                break;
            }
            //dbg!(candidate_vertexes.len());
            trace!("There are {} candidate vertexes", candidate_vertexes.len());
            graph_has_been_modified_this_iteration = false;
            for v in candidate_vertexes.drain(..) {
                self.remove_vertex(&v);
                graph_has_been_modified = true;
                graph_has_been_modified_this_iteration = true;
            }

            if !graph_has_been_modified_this_iteration {
                trace!("End of loop, and no changes made → break out");
                break;
            }
        }

        graph_has_been_modified
    }
}

pub trait DirectedGraphTrait: Send {
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

    /// Adds an edge between these 2, returning true iff the edge already existed
    fn add_edge(&mut self, vertex1: i64, vertex2: i64) -> bool;

    fn out_neighbours(&self, from_vertex: i64) -> impl Iterator<Item = i64>;

    fn num_vertexes(&self) -> usize;
    fn num_edges(&self) -> usize;
    /// True iff this vertex is in this graph
    fn contains_vertex(&self, vid: &i64) -> bool;

    /// Iterator over all edges
    fn edges_iter(&self) -> impl Iterator<Item = (i64, i64)>;

    /// returns each vertex and the number of out edges
    fn vertexes_and_num_outs(&self) -> impl Iterator<Item = (i64, usize)>;

    fn len(&self) -> (usize, usize) {
        (self.num_vertexes(), self.num_edges())
    }

    /// Iterator (in any order) of vertexes which are the destination of an edge
    fn dest_vertexes_jumbled(&self) -> impl Iterator<Item = i64> {
        self.edges_iter().map(|(_src, dest)| dest)
    }
    /// Iterator (in any order) of vertexes which are the src of an edge
    fn src_vertexes_jumbled(&self) -> impl Iterator<Item = i64> {
        self.edges_iter().map(|(src, _dest)| src)
    }

    /// True iff this vertex has an outgoing edge
    fn vertex_has_outgoing(&self, vid: &i64) -> bool;

    fn detailed_size(&self) -> String;

    /// Iterator (in any order, possibly with dupes) of vertexes which do not have outgoing edges
    fn vertexes_wo_outgoing_jumbled(&self) -> impl Iterator<Item = i64> {
        self.dest_vertexes_jumbled()
            .filter(|v| !self.vertex_has_outgoing(v))
    }
}

pub type SmallNidVec = SmallVec<[i64; 1]>;

/// A graph which stores a list of all incoming and outgoing edges
#[derive(Debug, Clone)]
pub struct DirectedGraph2 {
    // key is vertex id
    // value.0 is list of incoming vertexes  (ie there's an edge from something → key)
    // value.1 is list of outgoing vertexes (ie there's an edge from key → something)
    edges: BTreeMapSplitKey<(SmallNidVec, SmallNidVec)>,
}

impl DirectedGraph2 {
    pub fn new() -> Self {
        Self {
            edges: BTreeMapSplitKey::new(),
        }
    }

    /// returns each vertex and the number of in & out edges
    pub fn vertexes_and_num_ins_outs(&self) -> impl Iterator<Item = (i64, usize, usize)> + '_ {
        self.edges
            .iter()
            .map(|(v, (ins, outs))| (v, ins.len(), outs.len()))
    }

    /// What are the vertexes that we can reach *from* this vertex.
    /// i.e. the edges which go from `from_vertex`, where do they go?
    pub fn vertexes_reachable_from(&self, from_vertex: i64) -> impl Iterator<Item = i64> + '_ {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(|(_ins, outs)| outs.iter())
            .copied()
    }

    pub fn in_neighbours(&self, from_vertex: i64) -> impl Iterator<Item = i64> + '_ {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(|(ins, _outs)| ins.iter())
            .copied()
    }

    /// Number of in edges to this vertex. (None iff this vertex isn't in the graph)
    pub fn num_ins(&self, vid: &i64) -> Option<usize> {
        self.edges.get(vid).map(|(ins, _outs)| ins.len())
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Vertex v exists, and has either in neighbours, xor out neighbours (but not both)
    fn neighbors_in_xor_out(&mut self, v: &i64) -> bool {
        self.edges
            .get(v)
            .map_or(false, |(ins, outs)| !ins.is_empty() ^ !outs.is_empty())
    }

    pub fn remove_edge(&mut self, vertex1: &i64, vertex2: &i64) {
        if let Some(from_v1) = &mut self.edges.get_mut(vertex1).map(|(_to, from)| from) {
            from_v1.retain(|other| other != vertex2);
        }
        if self
            .edges
            .get(vertex1)
            .map_or(false, |(from, to)| from.is_empty() && to.is_empty())
        {
            self.edges.remove(vertex1);
        }
        if let Some(to_v2) = &mut self.edges.get_mut(vertex2).map(|(to, _from)| to) {
            to_v2.retain(|other| other != vertex1);
        }
        if self
            .edges
            .get(vertex2)
            .map_or(false, |(from, to)| from.is_empty() && to.is_empty())
        {
            self.edges.remove(vertex2);
        }
    }

    /// Removes this vertex (& associated edges) from this graph, and return the list of in- &
    /// out-edges that it was part of
    fn remove_vertex(&mut self, vertex: &i64) -> Option<(SmallNidVec, SmallNidVec)> {
        let (ins, outs) = match self.edges.remove(vertex) {
            None => {
                return None;
            }
            Some((ins, outs)) => (ins, outs),
        };

        // this could definitly be done more effeciently
        for other_src in ins.iter() {
            self.remove_edge(other_src, vertex);
        }
        for other_dest in outs.iter() {
            self.remove_edge(vertex, other_dest);
        }

        Some((ins, outs))
    }

    /// Remove any vertexes which have incoming edges, or outgoing edges, but not both.
    /// Vertexes like this cannot be part of a cycle. Removing them makes cycle detection faster.
    pub fn remove_vertexes_with_in_xor_out(&mut self) {
        let mut vertexes_to_remove = Vec::new();
        let mut round = 0;
        let (mut orig_num_vertexes, mut num_vertexes);

        let mut more_vertexes_to_remove = Vec::new();

        loop {
            orig_num_vertexes = self.num_vertexes();
            if orig_num_vertexes == 0 {
                break;
            }
            vertexes_to_remove.truncate(0);
            more_vertexes_to_remove.truncate(0);
            // Trying to make this parallel or sensible doesn't speed things up a lot...
            vertexes_to_remove.extend(self.edges.iter().filter_map(|(v, (ins, outs))| {
                if !ins.is_empty() ^ !outs.is_empty() {
                    Some(v)
                } else {
                    None
                }
            }));

            while !vertexes_to_remove.is_empty() {
                debug!(
                    "remove_vertexes_with_in_xor_out: round {round}. vertexes_to_remove.len() = {} num_vertexes {}", vertexes_to_remove.len(), self.num_vertexes());
                more_vertexes_to_remove.truncate(0);
                for v in vertexes_to_remove.drain(..) {
                    let ins_outs = self.remove_vertex(&v);
                    //removing_vertexes_bar.inc(1);
                    if let Some((ins, outs)) = ins_outs {
                        more_vertexes_to_remove
                            .extend(ins.into_iter().filter(|v| self.neighbors_in_xor_out(v)));
                        more_vertexes_to_remove
                            .extend(outs.into_iter().filter(|v| self.neighbors_in_xor_out(v)));
                    }
                }
                round += 1;
                std::mem::swap(&mut vertexes_to_remove, &mut more_vertexes_to_remove);
            }
            num_vertexes = self.num_vertexes();
            trace!(
                "remove_vertexes_with_in_xor_out: round {}. Now {} vertexes. {} have been removed",
                round,
                num_vertexes,
                orig_num_vertexes - num_vertexes
            );
            if num_vertexes == orig_num_vertexes {
                break;
            }
            round += 1;
        }
    }

    //pub fn take_possibly_one_cycle(
    //    &mut self,
    //    removing_vertexes_bar: &ProgressBar,
    //) -> Option<Vec<i64>> {
    //    // first remove any vertexes which only have incoming or ourgoing edges.
    //    if self.is_empty() {
    //        return None;
    //    }

    //    let possible_vertex = self.vertexes().next().unwrap().to_owned();

    //    match dij::look_for_loop(possible_vertex, self) {
    //        None => {
    //            self.remove_vertex(&possible_vertex);
    //            removing_vertexes_bar.inc(1);
    //            None
    //        }
    //        Some(cycle) => {
    //            for v in cycle.iter() {
    //                self.remove_vertex(v);
    //                removing_vertexes_bar.inc(1);
    //            }
    //            Some(cycle)
    //        }
    //    }
    //}

    //pub fn into_cycles(
    //    mut self,
    //    removing_vertexes_bar: &ProgressBar,
    //    cycles_found: &ProgressBar,
    //) -> Vec<Vec<i64>> {
    //    let mut round = 0;
    //    debug!("into_cycles: Starting. Graph has size: {:?}", self.len());
    //    let mut cycles = Vec::new();

    //    loop {
    //        round += 1;
    //        if self.is_empty() {
    //            break;
    //        }
    //        let old_len = self.len();

    //        // first remove any vertexes which only have incoming or ourgoing edges.
    //        self.remove_vertexes_with_in_xor_out();
    //        if self.is_empty() {
    //            break;
    //        }
    //        debug!("into_cycles: round {round}, removed {} vertexes which are irrelevant, leaving {} vertexes", (old_len.0 - self.len().0), self.num_vertexes());

    //        let possible_vertex = self.vertexes().next().unwrap().to_owned();
    //        let cycle = dij::look_for_loop(possible_vertex, &self);
    //        match cycle {
    //            None => {
    //                debug!(
    //                    "into_cycles: round {round}: {possible_vertex} isn't in a cycle. Removing it."
    //                );
    //                self.remove_vertex(&possible_vertex);
    //                removing_vertexes_bar.inc(1);
    //                continue; // back to loop
    //            }
    //            Some(cycle) => {
    //                debug!(
    //                    "into_cycles: round {round}: Found a cycle of length {}. Start nodeid {}",
    //                    cycle.len(),
    //                    cycle[0],
    //                );
    //                for v in cycle.iter() {
    //                    self.remove_vertex(v);
    //                    removing_vertexes_bar.inc(1);
    //                }
    //                cycles_found.inc(1);
    //                cycles.push(cycle);
    //            }
    //        }
    //    }

    //    debug!("Found {} cycles", cycles.len());

    //    cycles
    //}

    //// Do many Dij searches and return the first cycle we find
    //// start with the vertex id > start_vertex_id if set
    //fn find_one_cycle(&self, start_vertex_id: Option<i64>) -> Option<Vec<i64>> {
    //    match start_vertex_id {
    //        None => {
    //            for vertex in self.vertexes() {
    //                if let Some(cycle) = dij::look_for_loop(*vertex, self) {
    //                    return Some(cycle);
    //                }
    //            }
    //            None
    //        }
    //        Some(start_vertex_id) => {
    //            let vertexes = self.vertexes_range(start_vertex_id..);
    //            let vertexes = vertexes.chain(self.vertexes_range(..=start_vertex_id));

    //            let mut vertexes = vertexes.filter_map(|v| dij::look_for_loop(*v, self));

    //            vertexes.next()
    //        }
    //    }
    //}

    pub fn into_disconnected_graphs(self) -> impl Iterator<Item = DirectedGraph2> {
        let mut g = self;
        let mut vertexes_to_look_at = Vec::new();

        std::iter::from_fn(move || {
            if g.is_empty() {
                return None;
            }
            let mut new_graph = DirectedGraph2::new();
            vertexes_to_look_at.truncate(0);
            vertexes_to_look_at.push(g.vertexes().next().unwrap());

            while let Some(vertex) = vertexes_to_look_at.pop() {
                if let Some((ins, outs)) = g.remove_vertex(&vertex) {
                    for in_vertex in ins.into_iter() {
                        new_graph.add_edge(in_vertex, vertex);
                        vertexes_to_look_at.push(in_vertex);
                    }
                    for out_vertex in outs.into_iter() {
                        new_graph.add_edge(vertex, out_vertex);
                        vertexes_to_look_at.push(out_vertex);
                    }
                }
            }

            Some(new_graph)
        })
    }

    pub fn strongly_connected_components(
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

        //components.into_values().map(|v| v.to_vec()).collect()
    }

    //fn into_undirected_graph(self) -> UniDirectedGraph {
    //    let mut result = UniDirectedGraph::new();

    //    let DirectedGraph2 { edges } = self;
    //    for (v, (_ins, outs)) in edges.into_iter() {
    //        for out in outs.into_iter() {
    //            result.add_edge(v, out);
    //        }
    //    }

    //    result
    //}

    pub fn into_vertexes_topologically_sorted(self, sorting_nodes_bar: &ProgressBar) -> Vec<i64> {
        let mut g = self; //.clone();
        let mut result = Vec::with_capacity(g.num_vertexes());
        let mut no_incoming: BTreeSet<i64> = BTreeSet::new();

        let mut others = SmallNidVec::new();
        loop {
            no_incoming.extend(
                g.vertexes_and_num_ins_outs().filter_map(
                    |(v, num_ins, _num_outs)| if num_ins == 0 { Some(v) } else { None },
                ),
            );
            if no_incoming.is_empty() {
                break;
            }

            while let Some(v) = no_incoming.pop_first() {
                result.push(v);
                sorting_nodes_bar.inc(1);

                // have to save to another Vec to prevent lifetimes
                others.truncate(0);
                others.extend(g.vertexes_reachable_from(v));
                for other in others.drain(..) {
                    g.remove_edge(&v, &other);
                    if g.num_ins(&other).map_or(false, |n| n == 0) {
                        no_incoming.insert(other);
                    }
                }
            }
        }
        //dbg!(g.len());

        result
    }

    /// Iterator over all vertexes
    pub fn vertexes(&self) -> impl Iterator<Item = i64> + '_ {
        self.edges.keys()
    }
}

impl DirectedGraphTrait for DirectedGraph2 {
    /// Adds an edge between these 2, returning true iff the edge already existed
    fn add_edge(&mut self, vertex1: i64, vertex2: i64) -> bool {
        if vertex1 == vertex2 {
            return false;
        }
        let from_v1 = &mut self.edges.entry(vertex1).or_default().1;
        if from_v1.contains(&vertex2) {
            return true;
        } else {
            from_v1.push(vertex2);
            from_v1.sort();
        }

        // assume we never get inconsistant
        self.edges.entry(vertex2).or_default().0.push(vertex1);
        self.edges.entry(vertex2).or_default().0.sort();
        false
    }

    fn out_neighbours(&self, from_vertex: i64) -> impl Iterator<Item = i64> + '_ {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(|(_ins, outs)| outs.iter())
            .copied()
    }

    fn num_vertexes(&self) -> usize {
        self.edges.len()
    }
    fn num_edges(&self) -> usize {
        self.edges
            .iter()
            .map(|(_vertex, (_in_edges, out_edges))| out_edges.len())
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
            .flat_map(move |(v, (_ins, outs))| outs.iter().map(move |o| (v, *o)))
    }

    /// returns each vertex and the number of out edges
    fn vertexes_and_num_outs(&self) -> impl Iterator<Item = (i64, usize)> {
        self.edges.iter().map(|(v, (_ins, outs))| (v, outs.len()))
    }

    fn vertex_has_outgoing(&self, vid: &i64) -> bool {
        self.edges
            .get(vid)
            .is_some_and(|(_ins, outs)| !outs.is_empty())
    }

    fn detailed_size(&self) -> String {
        let mut s = format!(
            "DirectedGraph2: num_vertexes {} num_edges {}",
            self.num_vertexes(),
            self.num_edges(),
        );
        s.push_str(&format!(
            "\nSize of graph: {} = {} bytes.\nbytes/vertex = {:>.5}\nbytes/edge = {:>.5}",
            self.get_size(),
            self.get_size().to_formatted_string(&Locale::en),
            self.get_size() as f64 / self.num_vertexes() as f64,
            self.get_size() as f64 / self.num_edges() as f64,
        ));

        s
    }
}

/// A graph which stores a list of only the outgoing edges
/// Practically the same as the DirectedGraph2, but exists to use less memory for later processing
/// steps when we don't need the in edges of a graph
#[derive(Debug, Clone, Default)]
pub struct UniDirectedGraph {
    // key is vertex id
    // value is list of outgoing vertexes (ie there's an edge from key → something)
    edges: BTreeMapSplitKey<SmallNidVec>,
}

impl UniDirectedGraph {
    pub fn new() -> Self {
        UniDirectedGraph {
            edges: BTreeMapSplitKey::new(),
        }
    }
}

impl DirectedGraphTrait for UniDirectedGraph {
    /// Adds an edge between these 2, returning true iff the edge already existed
    fn add_edge(&mut self, vertex1: i64, vertex2: i64) -> bool {
        if vertex1 == vertex2 {
            return false;
        }
        // assume we never get inconsistant
        self.edges.entry(vertex1).or_default().push(vertex2);
        self.edges.get_mut(&vertex1).unwrap().sort();
        false
    }

    fn out_neighbours(&self, from_vertex: i64) -> impl Iterator<Item = i64> {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(|outs| outs.iter())
            .copied()
    }

    fn num_vertexes(&self) -> usize {
        self.edges.len()
    }
    fn num_edges(&self) -> usize {
        self.edges
            .iter()
            .map(|(_vertex, out_edges)| out_edges.len())
            .sum()
    }
    /// True iff this vertex is in this graph
    fn contains_vertex(&self, vid: &i64) -> bool {
        self.edges.contains_key(vid)
    }

    /// Iterator over all edges
    fn edges_iter(&self) -> impl Iterator<Item = (i64, i64)> + '_ {
        self.edges
            .iter()
            .flat_map(|(v, outs)| outs.iter().map(move |o| (v, *o)))
    }

    /// returns each vertex and the number of out edges
    fn vertexes_and_num_outs(&self) -> impl Iterator<Item = (i64, usize)> + '_ {
        self.edges.iter().map(|(v, outs)| (v, outs.len()))
    }

    fn vertex_has_outgoing(&self, vid: &i64) -> bool {
        self.edges.contains_key(vid)
    }

    fn detailed_size(&self) -> String {
        let mut s = format!(
            "UndirectedAdjGraph: num_vertexes {} num_edges {}",
            self.num_vertexes(),
            self.num_edges(),
        );
        s.push_str(&format!(
            "\nSize of graph: {} = {} bytes.\nbytes/vertex = {:>.5}\nbytes/edge = {:>.5}",
            self.get_size(),
            self.get_size().to_formatted_string(&Locale::en),
            self.get_size() as f64 / self.num_vertexes() as f64,
            self.get_size() as f64 / self.num_edges() as f64,
        ));

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test1() {
        let mut g = DirectedGraph2::new();
        g.add_edge(1, 100);
        assert_eq!(g.edges_iter().collect::<Vec<_>>(), vec![(1, 100)]);
        g.add_edge(1, 100);
        assert_eq!(g.edges_iter().collect::<Vec<_>>(), vec![(1, 100)]);
        g.add_edge(1, 2);
        assert_eq!(g.edges_iter().collect::<Vec<_>>(), vec![(1, 2), (1, 100)]);
        g.add_edge_chain(&[2, 3, 4, 10, 6]);
        assert_eq!(
            g.edges_iter().collect::<Vec<_>>(),
            vec![(1, 2), (1, 100), (2, 3), (3, 4), (4, 10), (10, 6)]
        );
    }
}
