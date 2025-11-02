#![allow(warnings)]
#![allow(dead_code, unused_imports)]
use super::*;
use geo::algorithm::convex_hull::qhull::quick_hull;
use geo::{
    CoordsIter,
    algorithm::convex_hull::ConvexHull,
    geometry::{Coord, MultiPoint, Point},
};
use graph::Graph2;
use graph::UndirectedAdjGraph;
use haversine::haversine_m_fpair;
use inter_store::InterStore;
use ordered_float::OrderedFloat;
use sorted_slice_store::SortedSliceSet;
use std::collections::HashSet;

#[derive(Debug, Default)]
pub struct WayGroup {
    pub graph: Graph2,
    pub length_m: f64,
    pub json_props: serde_json::Value,
    pub group: Box<[Option<String>]>,
    pub root_nodeid: i64,
}

impl WayGroup {
    pub fn new(graph: Graph2, group: Box<[Option<String>]>) -> Self {
        let root_nodeid = *graph.first_vertex().unwrap();
        WayGroup {
            graph,
            root_nodeid,
            group,
            ..Default::default()
        }
    }

    pub fn calculate_length(&mut self, nodeid_pos: &impl NodeIdPosition) {
        self.length_m = self
            .graph
            .edges_par_iter()
            .map(|(a, b)| haversine_m_fpair(nodeid_pos.get(a).unwrap(), nodeid_pos.get(b).unwrap()))
            .sum::<f64>();
    }

    pub fn num_nodes(&self) -> usize {
        self.graph.num_vertexes()
    }

    pub fn filename(&self, output_filename: &str, split_files_by_group: bool) -> String {
        if !split_files_by_group {
            output_filename.to_string()
        } else {
            output_filename.replace(
                "%s",
                &self
                    .group
                    .iter()
                    .map(|s| {
                        s.as_ref()
                            .map_or_else(|| "null".to_string(), |v| v.replace('/', "%2F"))
                    })
                    .collect::<Vec<_>>()
                    .join(","),
            )
        }
    }
    #[allow(unused)]
    pub fn recalculate_root_id(&mut self) {
        self.root_nodeid = *self.graph.first_vertex().unwrap_or(&0);
    }

    /// Calculate the frames for this way group
    pub fn frames(
        &self,
        nodeid_pos: &impl NodeIdPosition,
        frames_bar: &ProgressBar,
    ) -> impl Iterator<Item = Box<[i64]>> {
        // First calculate the
        let mut all_nodes_pos = self
            .graph
            .vertexes_par_iter()
            .map(|nid| nodeid_pos.get(nid).unwrap())
            .collect::<Vec<_>>();
        all_nodes_pos.par_sort_unstable_by_key(|(x, y)| (OrderedFloat(*x), OrderedFloat(*y)));
        all_nodes_pos.dedup();
        let mut all_nodes_pos = all_nodes_pos
            .into_iter()
            .map(|c| Coord { x: c.0, y: c.1 })
            .collect::<Vec<_>>();

        let chull = quick_hull(&mut all_nodes_pos).into_inner();

        drop(all_nodes_pos);
        let mut chull: Vec<(OrderedFloat<f64>, OrderedFloat<f64>)> = chull
            .into_iter()
            .map(|c| (c.x.into(), c.y.into()))
            .collect();
        chull.par_sort_by_key(|(x, y)| (OrderedFloat(*x), OrderedFloat(*y)));
        chull.dedup();
        let n = chull.len() as u64;
        frames_bar.inc_length((n * (n - 1) / 2) * self.graph.num_vertexes() as u64);
        let chull = SortedSliceSet::from_vec(chull);
        //furthest_path_bar.inc_length((chull.len() * (chull.len() + 1) / 2) as u64);

        // We need the nodeids of these positions.
        // This is very effecient, we're basically doing a nodeid_pos lookup *again*.
        let mut convex_hull_nodes: Vec<_> = self
            .graph
            .vertexes_par_iter()
            .filter_map(|nid| {
                let p = nodeid_pos.get(nid).unwrap();
                let p = (p.0.into(), p.1.into());
                if chull.contains(&p) {
                    Some((*nid, p))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(convex_hull_nodes.len(), chull.len());
        drop(chull);
        convex_hull_nodes.par_sort_by_key(|(n, _)| *n);

        // Frames have a lot of overlap with each other.
        let mut frames_graph = Arc::new(Mutex::new(Graph2::new()));

        convex_hull_nodes
            .par_iter()
            .enumerate()
            .flat_map(|(i, source)| {
                // returns an iterator for each of the other nodes
                dij::paths_one_to_many(
                    *source,
                    &convex_hull_nodes[(i + 1)..],
                    nodeid_pos,
                    &self.graph,
                )
            })
            .inspect(|_| frames_bar.inc(self.graph.num_vertexes() as u64))
            .for_each_with(
                frames_graph.clone(),
                |frames_graph, ((from_nid, to_nid), path)| {
                    frames_graph.lock().unwrap().add_edge_chain(&path);
                },
            );

        let mut frames_graph = Arc::try_unwrap(frames_graph).unwrap().into_inner().unwrap();

        frames_graph.into_lines_random()
    }

    pub fn coords<'a>(
        &'a self,
        nodeid_pos: &'a impl NodeIdPosition,
    ) -> impl Iterator<Item = Vec<(f64, f64)>> + 'a {
        // very simple, just each edge
        self.graph
            .edges_iter()
            .map(|(a, b)| vec![nodeid_pos.get(a).unwrap(), nodeid_pos.get(b).unwrap()])
    }

    pub fn into_props_coords_random(
        self,
        nodeid_pos: &impl NodeIdPosition,
        inter_store: &InterStore,
    ) -> (serde_json::Value, Vec<Vec<(f64, f64)>>) {
        let Self {
            json_props, graph, ..
        } = self;
        let coords = graph
            .into_lines_random()
            .map(|line| {
                inter_store
                    .expand_line_undirected(&line)
                    .map(|nid| nodeid_pos.get(&nid).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect();
        (json_props, coords)
    }
}

impl PartialEq for WayGroup {
    fn eq(&self, other: &Self) -> bool {
        self.root_nodeid == other.root_nodeid
    }
}
impl Eq for WayGroup {}

impl PartialOrd for WayGroup {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for WayGroup {
    fn cmp(&self, other: &Self) -> Ordering {
        self.length_m.total_cmp(&other.length_m).reverse()
    }
}
