use super::*;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use rayon::prelude::ParallelIterator;

pub(crate) struct UndirectedGraph<T>
where
    T: Clone,
{
    data: Vec<T>,
    size: usize,
}

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

pub(crate) struct UndirectedAdjGraph<V, E>
{
    edges: BTreeMap<V, BTreeMap<V, (E, Vec<V>)>>,
}

impl<V, E> UndirectedAdjGraph<V, E>
where
    V: std::hash::Hash + Eq + Copy + Ord + Send + std::fmt::Debug,
    E: Copy + Clone + std::fmt::Debug + std::ops::Add<Output=E> + std::cmp::PartialEq,
{
    pub fn new() -> Self {
        Self {
            edges: BTreeMap::new(),
        }
    }

    pub fn set(&mut self, i: &V, j: &V, val: E) {
        self.edges
            .entry(*i)
            .or_default()
            .insert(*j, (val.clone(), vec![]));
        self.edges.entry(*j).or_default().insert(*i, (val, vec![]));
    }

    pub fn get(&self, i: &V, j: &V) -> Option<&E> {
        self.edges
            .get(&i)
            .and_then(|from_i| from_i.get(&j).map(|(e, intermediates)| e))
    }

    pub fn get_all(&self, i: &V, j: &V) -> Option<&(E, Vec<V>)> {
        self.edges
            .get(&i)
            .and_then(|from_i| from_i.get(&j))
    }

    pub fn get_intermediates(&self, i: &V, j: &V) -> Option<&[V]> {
        self.get_all(i, j).and_then(|(_e, intermediates)| Some(intermediates.as_slice()))
    }

    /// returns each vertex id and how many neighbours it has
    pub fn iter_vertexes_num_neighbours(&self) -> impl Iterator<Item=(&V, usize)> {
        self.edges.iter().map(|(vid, edges)| (vid, edges.len()))
    }

    pub fn contains_vertex(&self, v: &V) -> bool {
        self.edges.contains_key(v)
    }

    /// All the neighbours of this vertex and the edge weight
    pub fn neighbors(&self, i: &V) -> impl Iterator<Item = (&V, &E)> {
        self.edges[i]
            .iter()
            .map(|(j, (edge_weight, intermediates))| (j, edge_weight))
    }
    /// Number of neighbours for this vertex. 
    pub fn num_neighbors(&self, i: &V) -> usize {
        self.edges.get(i).map_or(0, |es| es.len())
    }

    pub fn max_vertex_id(&self) -> V {
        self.edges.keys().max().unwrap().clone()
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
    pub fn pretty_print(&self, fmt: &dyn Fn(&E) -> String, col_join: &str) -> String {
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
        if ! self.contains_vertex(v) {
            warn!("Called contract_vertex on v: {:?} which doesn't exist", v);
            return false;
        }
        if self.num_neighbors(v) != 2 {
            trace!("Called contract_vertex on v: {:?} and it has {} ≠ 2 neighbours", v, self.num_neighbors(v));
            return false;
        }
        // a - v - b
        let a = self.edges[v].keys().nth(0).unwrap().clone();
        let b = self.edges[v].keys().nth(1).unwrap().clone();
        assert!(a != b);
        if self.edges[&a].contains_key(&b) {
            // there already is an edge from a↔b, so skip this
            //trace!("v:{:?} There already is an edge from a-b (a={:?} b={:?})", v, a, b);
            return false;
        }
        assert!(self.edges[&a].contains_key(v));
        assert!(self.edges[&b].contains_key(v));
        assert!(self.edges[&a][v].0+self.edges[v][&b].0 == self.edges[&b][v].0+self.edges[&v][&a].0);
        let mut edge_a_v = self.edges.get_mut(&a).unwrap().remove(v).unwrap();
        let mut edge_b_v = self.edges.get_mut(&b).unwrap().remove(v).unwrap();
        let mut edge_v_a = self.edges.get_mut(v).unwrap().remove(&a).unwrap();
        let mut edge_v_b = self.edges.get_mut(v).unwrap().remove(&b).unwrap();
        assert!(self.edges[v].is_empty());
        self.edges.remove(v);
        let new_weight = edge_a_v.0 + edge_v_b.0;
        let mut a_b_intermediates: Vec<V> = vec![];
        a_b_intermediates.extend(edge_a_v.1.drain(..));
        a_b_intermediates.push(v.clone());
        a_b_intermediates.extend(edge_v_b.1.drain(..));
        let new_edge_a_b = (new_weight, a_b_intermediates);
        let mut new_edge_b_a = new_edge_a_b.clone();
        new_edge_b_a.1.reverse();

        self.edges.get_mut(&a).unwrap().insert(b, new_edge_a_b);
        self.edges.get_mut(&b).unwrap().insert(a, new_edge_b_a);

        return true;
    }

    pub fn contract_edges(&mut self) {
        let initial_num_edges = self.num_edges();
        let initial_num_vertexes = self.num_vertexes();
        trace!("Starting contract_edges with {} edges and {} vertexes", initial_num_edges, initial_num_vertexes);
        if initial_num_edges == 1 {
            return;
        }

        let mut graph_has_been_modified = false;
        let mut candidate_vertexes = Vec::new();
        let mut contraction_round = 0;
        let mut this_vertex_contracted = false;
        loop {
            trace!("Contraction round {}. There are {} vertexes and {} edges", contraction_round, self.num_vertexes(), self.num_edges());
            contraction_round+=1;
            candidate_vertexes.extend(self.iter_vertexes_num_neighbours().filter_map(|(v, nn)| if nn == 2 { Some(v) } else { None }).cloned());
            if candidate_vertexes.is_empty() {
                trace!("No more candidate vertexes");
                break;
            }
            trace!("There are {} candidate vertexes", candidate_vertexes.len());
            graph_has_been_modified = false;
            for v in candidate_vertexes.drain(..) {
                this_vertex_contracted = self.contract_vertex(&v);
                if this_vertex_contracted {
                    //trace!("Vertex {:?} was contracted", v);
                    graph_has_been_modified = true;
                } else {
                    //trace!("Vertex {:?} was not contracted", v);
                }
            }

            if ! graph_has_been_modified {
                trace!("End of loop, and no changes made → break out");
                break;
            }
        }

        debug!("End of contract_edges there are now {} edges and {} vertexes. Removed {} edges and {} vertexes in {} rounds", self.num_edges(), self.num_vertexes(), initial_num_edges-self.num_edges(), initial_num_vertexes-self.num_vertexes(), contraction_round);

    }
}
