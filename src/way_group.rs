use super::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
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
            root_wayid: root_wayid.into(),
            way_ids: vec![],
            nodeids: vec![],
            length_m: None,
            coords: None,
            extra_json_props: serde_json::from_str("{}").unwrap(),
            group: group,
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

    pub fn set_coords(&mut self, nodeid_pos: &HashMap<i64, (f64, f64)>) {
        if self.coords.is_some() {
            return;
        }
        self.coords = Some(
            self.nodeids
                .par_iter()
                .map(|nids| {
                    nids.par_iter()
                        .map(|nid| nodeid_pos.get(nid).map_or_else(
                                || {
                                    error!("Cannot find position for node id {}, way_group root_wayid {}. Skipping this node", nid, self.root_wayid);
                                    None
                                },
                                |p| Some(p.to_owned())
                                )
                            )
                        .filter_map(|p| p)
                        .collect::<Vec<(f64, f64)>>()
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
    pub fn recalculate_root_id(&mut self) {
        self.root_wayid = *self
            .nodeids
            .par_iter()
            .flat_map(|ns| ns.par_iter())
            .min()
            .unwrap_or(&0);
    }

    pub fn distance_m(&self, other: &WayGroup) -> Option<f64> {
        if self.coords.is_none() || other.coords.is_none() {
            return None;
        }

        // Attempted shortcut. If they share a nodeid, then shortest distance is 0
        if self
            .nodeids_iter()
            .any(|n1| other.nodeids_iter().any(|n2| n1 == n2))
        {
            return Some(0.);
        }

        self.coords
            .as_ref()
            .unwrap()
            .par_iter()
            .flat_map(|cs| cs.par_iter())
            .flat_map(|c1| {
                other
                    .coords
                    .as_ref()
                    .unwrap()
                    .par_iter()
                    .flat_map(|cs| cs.par_iter())
                    .map(move |c2| (c1, c2))
            })
            .map(|(c1, c2)| haversine_m(c1.0, c1.1, c2.0, c2.1))
            .min_by(|d1, d2| d1.total_cmp(&d2))
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
        match (self.length_m, other.length_m) {
            (Some(a), Some(b)) => Some(a.total_cmp(&b).reverse()),
            _ => Some(self.root_wayid.cmp(&other.root_wayid)),
        }
    }
}
impl Ord for WayGroup {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
