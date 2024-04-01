#![allow(dead_code)]
use super::*;

#[derive(Debug, Clone, Default)]
pub struct WayGroup {
    pub root_wayid: i64,
    pub way_ids: Vec<i64>,
    pub nodeids: Vec<Vec<i64>>,
    pub length_m: Option<f64>,
    pub coords: Option<Vec<Vec<(f64, f64)>>>,
    pub extra_json_props: serde_json::Value,
    pub group: Vec<Option<String>>,
}

impl WayGroup {
    pub fn new(root_wayid: impl Into<i64>, group: Vec<Option<String>>) -> Self {
        WayGroup {
            group,
            root_wayid: root_wayid.into(),
            extra_json_props: serde_json::from_str("{}").unwrap(),
            ..Default::default()
        }
    }

    pub fn calculate_length(&mut self) {
        if self.length_m.is_some() {
            return;
        }
        self.length_m = Some(
            self.coords
                .as_ref()
                .unwrap()
                .par_iter()
                .map(|coord_string| {
                    coord_string
                        .par_windows(2)
                        .map(|pair| haversine_m(pair[0].1, pair[0].0, pair[1].1, pair[1].0))
                        .sum::<f64>()
                })
                .sum(),
        )
    }

    pub fn set_coords(&mut self, nodeid_pos: &impl NodeIdPosition) {
        if self.coords.is_some() {
            return;
        }
        self.coords = Some(
            self.nodeids
                .par_iter()
                .map(|nids| {
                    let mut poses = vec![(-200., -200.); nids.len()];
                    nodeid_pos.get_many_unwrap(nids, poses.as_mut_slice());
                    poses
                })
                .collect::<Vec<_>>(),
        );
    }

    pub fn num_nodeids(&self) -> usize {
        self.nodeids.par_iter().map(|nids| nids.len()).sum()
    }

    pub fn nodeids_iter(&self) -> impl rayon::prelude::ParallelIterator<Item = &i64> + '_ {
        self.nodeids.par_iter().flat_map(|nids| nids.par_iter())
    }

    pub fn coords_iter_par(&self) -> impl rayon::prelude::ParallelIterator<Item = [f64; 2]> + '_ {
        self.coords
            .as_ref()
            .expect("You called WayGroup::coords_iter_par before you have set the coords for this waygroup")
            .par_iter()
            .flat_map(|coord_string| coord_string.par_iter().map(|c| [c.0, c.1]))
    }

    pub fn coords_iter_seq(&self) -> impl Iterator<Item = [f64; 2]> + '_ {
        //pub coords: Option<Vec<Vec<(f64, f64)>>>,
        self.coords
            .as_ref()
            .expect("You called WayGroup::coords_iter_seq before you have set the coords for this waygroup")
            .iter()
            .flat_map(|coord_string| coord_string.iter().map(|c| [c.0, c.1]))
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
        self.root_wayid = *self
            .nodeids
            .par_iter()
            .flat_map(|ns| ns.par_iter())
            .min()
            .unwrap_or(&0);
    }

    /// Try to reduce the number of inner segments, by merging segments which are end to end
    /// connected
    pub fn reorder_segments(
        &mut self,
        max_rounds: impl Into<Option<usize>>,
        reorder_segments_bar: &ProgressBar,
        can_reverse_ways: bool,
    ) {
        let max_rounds = max_rounds.into();
        let old_num_nodeids = self.nodeids.len();
        if old_num_nodeids == 1 {
            // nothing to do
            reorder_segments_bar.inc(self.nodeids.len() as u64);
            return;
        }
        trace!(
            "wg:{} Before reorder_segments there are {old_num_nodeids} segments",
            self.root_wayid,
        );

        let mut graph_modified;
        let mut round = 0;
        let (mut seg_i, mut seg_j);
        let mut num_nodes;
        let (mut left, mut right);
        let (mut i, mut j);

        let mut nid_num_neighbours: HashMap<i64, Vec<usize>> =
            HashMap::with_capacity(2 * self.nodeids.len());

        // Main loop to remove segments
        loop {
            if self.nodeids.len() == 1 {
                break;
            }
            // shortest segments first. this seems to have the largest reduction in segments
            self.nodeids.par_sort_by_key(|e| e.len());
            graph_modified = false;
            num_nodes = self.nodeids.len();
            if old_num_nodeids > 1_000 {
                trace!(
                    "wg:{} reorder_segments. round {round}. There are {num_nodes} nodeids",
                    self.root_wayid
                );
            }
            if max_rounds.map_or(false, |max| round >= max) {
                trace!(
                    "wg:{} reorder_segments. round {round}. Reached max rounds, breaking out",
                    self.root_wayid
                );
                break;
            }
            round += 1;

            nid_num_neighbours.clear();
            for (i, segment) in self.nodeids.iter().enumerate() {
                if segment.is_empty() {
                    continue;
                }
                nid_num_neighbours
                    .entry(*segment.first().unwrap())
                    .or_default()
                    .push(i);
                nid_num_neighbours
                    .entry(*segment.last().unwrap())
                    .or_default()
                    .push(i);
            }
            nid_num_neighbours.shrink_to_fit();

            for edges in nid_num_neighbours
                .drain()
                .map(|(_k, v)| v)
                .filter(|e| e.len() >= 2)
            {
                i = edges[0];
                j = edges[1];
                if i == j {
                    continue;
                }
                if i > j {
                    std::mem::swap(&mut i, &mut j);
                }
                (left, right) = self.nodeids.split_at_mut(i + 1);
                seg_i = left.last_mut().unwrap();
                seg_j = right.get_mut(j - i - 1).unwrap();
                if seg_i.is_empty() || seg_j.is_empty() {
                    continue;
                }

                if seg_i.last() == seg_j.first() {
                    graph_modified = true;
                    reorder_segments_bar.inc(1);
                    seg_i.extend(seg_j.drain(..).skip(1));
                } else if can_reverse_ways && seg_i.last() == seg_j.last() {
                    graph_modified = true;
                    reorder_segments_bar.inc(1);
                    seg_i.extend(seg_j.drain(..).rev().skip(1));
                } else if seg_i.first() == seg_j.first() {
                    graph_modified = true;
                    reorder_segments_bar.inc(1);
                    seg_i.reverse();
                    seg_i.extend(seg_j.drain(..).skip(1));
                } else if can_reverse_ways && seg_i.first() == seg_j.last() {
                    graph_modified = true;
                    reorder_segments_bar.inc(1);
                    seg_i.reverse();
                    seg_i.extend(seg_j.drain(..).rev().skip(1));
                }
            }

            if !graph_modified {
                break;
            }
        }
        drop(nid_num_neighbours);
        // Shrink our total
        self.nodeids.retain(|segments| !segments.is_empty());

        self.nodeids.shrink_to_fit();
        // these segments are “done”
        reorder_segments_bar.inc(self.nodeids.len() as u64);

        // coords no longer valid
        self.coords = None;
        // in theory this shouldn't change, but just in case
        self.length_m = None;

        log!(
            if old_num_nodeids > 20_000 || old_num_nodeids - self.nodeids.len() > 20_000 {
                Debug
            } else {
                Trace
            },
            "wg:{} After reorder_segments there are {}K segments, removed {}K, in {round} round(s)",
            self.root_wayid,
            self.nodeids.len() / 1_000,
            old_num_nodeids.abs_diff(self.nodeids.len()) / 1_000,
        );
    }
}

impl PartialEq for WayGroup {
    fn eq(&self, other: &Self) -> bool {
        self.root_wayid == other.root_wayid
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
        match (self.length_m, other.length_m) {
            (Some(a), Some(b)) => a.total_cmp(&b).reverse(),
            _ => self.root_wayid.cmp(&other.root_wayid),
        }
    }
}
