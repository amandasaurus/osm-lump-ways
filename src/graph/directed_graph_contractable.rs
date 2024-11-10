use super::*;

// Store (list of nodes that are “in”. and then list of nodes that are out, along with the
// full intermediate chain.
// We only store a list of the
#[derive(Debug, Default)]
pub struct DirectedGraphContractable {
    edges: BTreeMap<i64, (SmallNidVec, BTreeMap<i64, Vec<i64>>)>,
}

impl DirectedGraphContractable {
    pub fn num_outs(&self, v: &i64) -> usize {
        self.edges.get(v).map_or(0, |o| o.1.len())
    }

    pub fn piller_edges(&self) -> impl Iterator<Item = (i64, i64)> + '_ {
        self.edges
            .iter()
            .flat_map(|(v1, (_in_vertexes, outs))| outs.iter().map(move |(v2, _inters)| (*v1, *v2)))
    }

    pub fn create_directed_graph_from_pillars(&self) -> DirectedGraph2 {
        let mut g = DirectedGraph2::new();
        for (a, b) in self.piller_edges() {
            g.add_edge(a, b);
        }
        g
    }

    pub fn create_intermediate_vertex_store(&self) -> IntermediateVertexStore<i64> {
        let mut st = IntermediateVertexStore::new();

        for (a, b, inters) in self.edges.iter().flat_map(|(a, (_in_vertexes, outs))| {
            outs.iter().map(move |(b, inters)| (a, b, inters))
        }) {
            st.add_next(&(*a, *b), inters);
        }

        st
    }

    fn num_pillar_edges(&self) -> usize {
        self.edges.len()
    }
    pub fn num_pillar_vertexes(&self) -> usize {
        self.edges.len()
    }

    pub fn remove_vertex_w_inters(
        &mut self,
        from_vertex: &i64,
        to_vertex: &i64,
    ) -> Option<Vec<i64>> {
        if !self
            .edges
            .get(from_vertex)
            .map_or(false, |(_ins, outs)| outs.contains_key(to_vertex))
        {
            return None;
        }
        self.edges
            .get_mut(to_vertex)
            .unwrap()
            .0
            .retain(|v| v != from_vertex);
        let tos = self.edges.get(to_vertex).unwrap();
        if tos.0.is_empty() && tos.1.is_empty() {
            self.edges.remove(to_vertex);
        }

        let inters = self
            .edges
            .get_mut(from_vertex)
            .unwrap()
            .1
            .remove(to_vertex)
            .unwrap();
        let froms = self.edges.get(from_vertex).unwrap();
        if froms.0.is_empty() && froms.1.is_empty() {
            self.edges.remove(from_vertex);
        }

        Some(inters)
    }

    pub fn add_edge_with_iters(&mut self, from_vertex: &i64, to_vertex: &i64, inters: Vec<i64>) {
        //if self.contains_edge(*from_vertex, *to_vertex) {
        //    return;
        //    //panic!("Can't double add, graph already has an edge from {} to {}. curr from: {:?} proposed inters: {:?}", from_vertex, to_vertex, self.edges[from_vertex], inters);
        //}
        self.edges
            .entry(*to_vertex)
            .or_default()
            .0
            .push(*from_vertex);
        self.edges
            .entry(*from_vertex)
            .or_default()
            .1
            .insert(*to_vertex, inters);
    }
}

impl ContractableDirectedGraph for DirectedGraphContractable {
    fn add_edge_contractable(
        &mut self,
        vertex1: i64,
        vertex2: i64,
        can_contract_vertex: &impl Fn(&i64) -> bool,
    ) -> bool {
        if vertex1 == vertex2 {
            return false;
        }
        if self.contains_edge(vertex1, vertex2) {
            return false;
        }

        let v1 = self.edges.entry(vertex1).or_default();
        v1.1.insert(vertex2, Default::default());
        self.edges.entry(vertex2).or_default().0.push(vertex1);

        if can_contract_vertex(&vertex1) {
            self.attempt_contract_vertex(&vertex1);
        }
        if can_contract_vertex(&vertex2) {
            self.attempt_contract_vertex(&vertex2);
        }
        true
    }
    fn attempt_contract_vertex(&mut self, vertex_id: &i64) -> bool {
        //return false;
        if !self.edges.contains_key(vertex_id) {
            return false;
        }
        let (in_edges, out_edges) = self.edges.get(vertex_id).unwrap();
        if !(in_edges.len() == 1 && out_edges.len() == 1) {
            return false;
        }
        // vertex_left → vertex_id → vertex_right
        let vertex_left = in_edges[0];
        let vertex_right = *out_edges.first_key_value().unwrap().0;
        if self.contains_edge(vertex_left, vertex_right) {
            // we already have a edge
            return false;
        }
        drop(in_edges);
        drop(out_edges);

        // everything OK, we can contract this vertex

        let inters_left_middle = self
            .remove_vertex_w_inters(&vertex_left, vertex_id)
            .unwrap();
        let inters_middle_right = self
            .remove_vertex_w_inters(vertex_id, &vertex_right)
            .unwrap();

        let mut inters_left_right = inters_left_middle;
        inters_left_right.push(*vertex_id);
        inters_left_right.extend(inters_middle_right);

        self.add_edge_with_iters(&vertex_left, &vertex_right, inters_left_right);

        true
    }
    fn attempt_contract_all(&mut self, can_contract_vertex: &impl Fn(&i64) -> bool) -> bool {
        let mut possible_vertexes = Vec::new();
        let graph_changed = false;

        loop {
            possible_vertexes.truncate(0);
            possible_vertexes.extend(
                self.edges
                    .iter()
                    .filter(|(_vertex_id, (in_edges, out_edges))| {
                        in_edges.len() == 1 && out_edges.len() == 1
                    })
                    .filter(|(vertex_id, _edges)| can_contract_vertex(vertex_id))
                    .map(|(vertex_id, _edges)| vertex_id),
            );
            if possible_vertexes.is_empty() {
                break;
            }

            let mut changed_this_loop = false;
            while let Some(v) = possible_vertexes.pop() {
                changed_this_loop |= self.attempt_contract_vertex(&v);
            }

            if !changed_this_loop {
                break;
            }
        }

        graph_changed
    }
}

impl DirectedGraphTrait for DirectedGraphContractable {
    fn new() -> Self {
        Self::default()
    }

    fn add_edge(&mut self, vertex1: i64, vertex2: i64) -> bool {
        self.add_edge_contractable(vertex1, vertex2, &|_| false);
        true
    }

    fn in_neighbours(&self, from_vertex: i64) -> impl Iterator<Item = i64> {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(move |(ins, _outs)| ins.iter().copied())
    }
    fn out_neighbours(&self, from_vertex: i64) -> impl Iterator<Item = i64> {
        self.edges
            .get(&from_vertex)
            .into_iter()
            .flat_map(move |(_in_list, outs)| outs.keys().copied())
    }

    fn num_vertexes(&self) -> usize {
        todo!()
    }
    /// True iff this vertex is in this graph
    fn contains_vertex(&self, _vid: &i64) -> bool {
        todo!()
    }

    #[allow(unreachable_code)]
    /// Iterator over all edges
    fn edges_iter(&self) -> impl Iterator<Item = (i64, i64)> + '_ {
        todo!();
        std::iter::empty()
    }
    #[allow(unreachable_code)]
    fn edges_par_iter(&self) -> impl ParallelIterator<Item = (i64, i64)> {
        todo!();
        Vec::new().into_par_iter()
    }

    #[allow(unreachable_code)]
    /// returns each vertex and the number of out edges
    fn vertexes_and_num_outs(&self) -> impl Iterator<Item = (i64, usize)> + '_ {
        todo!();
        std::iter::empty()
    }

    fn detailed_size(&self) -> String {
        todo!()
    }

    #[allow(unreachable_code)]
    /// returns each vertex and the number of in & out edges
    fn vertexes_and_num_ins_outs(&self) -> impl Iterator<Item = (i64, usize, usize)> + '_ {
        todo!();
        std::iter::empty()
    }

    /// Removes this vertex (& associated edges) from this graph, and return the list of in- &
    /// out-edges that it was part of
    fn remove_vertex(&mut self, _vertex: &i64) -> Option<(SmallNidVec, SmallNidVec)> {
        todo!()
    }

    #[allow(unreachable_code)]
    /// Iterator over all vertexes
    fn vertexes(&self) -> impl Iterator<Item = i64> + '_ {
        todo!();
        std::iter::empty()
    }

    #[allow(unreachable_code)]
    fn into_disconnected_graphs(self, _progress_bar: &ProgressBar) -> impl Iterator<Item = Self> {
        todo!();
        std::iter::empty()
    }
    fn remove_edge(&mut self, _vertex1: &i64, _vertex2: &i64) {
        todo!()
    }

    fn expand_edge(&self, vertex1: i64, vertex2: i64) -> impl Iterator<Item = i64> + '_ {
        self.edges
            .get(&vertex1)
            .and_then(move |(_ins, outs)| outs.get(&vertex2))
            .into_iter()
            .flat_map(move |inters| {
                iter::once(vertex1)
                    .chain(inters.iter().copied())
                    .chain(iter::once(vertex2))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod contractablegraph {
        use super::*;

        #[test]
        fn simple() {
            let mut g = DirectedGraphContractable::new();
            g.add_edge_contractable(1, 2, &|_| false);
            g.add_edge_contractable(2, 3, &|_| false);

            let pillar_edges = g.piller_edges().collect::<Vec<_>>();
            assert_eq!(pillar_edges, vec![(1, 2), (2, 3)]);
        }

        #[test]
        fn contract1() {
            let mut g = DirectedGraphContractable::new();
            let can_contract_vertex = g.add_edge_contractable(1, 2, &|&n| n == 2);
            g.add_edge_contractable(2, 3, &|&n| n == 2);
            g.attempt_contract_all(&|&n| n == 2);

            let pillar_edges = g.piller_edges().collect::<Vec<_>>();
            assert_eq!(pillar_edges, vec![(1, 3)]);
        }

        #[test]
        fn contract2() {
            let mut g = DirectedGraphContractable::new();
            g.add_edge_contractable(2, 3, &|&n| n == 2);
            g.add_edge_contractable(1, 2, &|&n| n == 2);
            // it will have contracted everything in place.

            let pillar_edges = g.piller_edges().collect::<Vec<_>>();
            assert_eq!(pillar_edges, vec![(1, 3)]);
        }

        #[test]
        fn contract_chain_live() {
            let mut g = DirectedGraphContractable::new();
            g.add_edge_chain_contractable(&[1, 2, 3], &|&n| n == 2);
            // it will have contracted everything in place.

            let pillar_edges = g.piller_edges().collect::<Vec<_>>();
            assert_eq!(pillar_edges, vec![(1, 3)]);
        }

        #[test]
        fn remove1() {
            let mut g = DirectedGraphContractable::new();
            g.add_edge_chain_contractable(&[1, 2, 3, 4], &|_| true);
            g.add_edge_chain_contractable(&[5, 6], &|_| true);

            let pillar_edges = g.piller_edges().collect::<Vec<_>>();
            assert_eq!(pillar_edges, vec![(1, 4), (5, 6)]);

            let res = g.remove_vertex_w_inters(&1, &4).unwrap();
            assert_eq!(res, vec![2, 3]);
            let pillar_edges = g.piller_edges().collect::<Vec<_>>();
            assert_eq!(pillar_edges, vec![(5, 6)]);
        }
    }

    mod rwindows2 {
        use super::*;

        macro_rules! test_rwindows2 {
            ( $name:ident, $input: expr, $expected_output:expr ) => {
                #[test]
                fn $name() {
                    let input: Vec<_> = $input;
                    let expected_output: Vec<_> = $expected_output;
                    let output: Vec<_> = rwindows2(&input).map(|(a, b)| (*a, *b)).collect();
                    assert_eq!(
                        output, expected_output,
                        "Ouput was {:?} but expected {:?}",
                        output, expected_output,
                    );
                }
            };
        }

        test_rwindows2!(simple1, vec![] as Vec<i64>, vec![]);
        test_rwindows2!(simple2, vec![1], vec![]);
        test_rwindows2!(simple3, vec![1, 2], vec![(1, 2)]);
        test_rwindows2!(simple4, vec![1, 2, 3], vec![(2, 3), (1, 2)]);
        test_rwindows2!(simple5, vec![1, 2, 3, 4], vec![(3, 4), (2, 3), (1, 2)]);
    }
}
