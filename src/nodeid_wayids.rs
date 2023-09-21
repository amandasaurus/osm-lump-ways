//! A struct for storing which wayid(s) a nodeid is in.
//! Most nodes are in only 1 way. This struct uses much less memory by taking advantage of that.
//! A small amount of nodes are in exactly 2 nodes (This saves about 10% space)
use super::*;
use std::collections::BTreeMap;
use std::fmt::Debug;
use vartyint;

/// Something which stores which nodeids are in which wayid
pub(crate) trait NodeIdWayIds: Debug + Send + Sync {
    fn new() -> Self
    where
        Self: Sized + Send;

    /// Number of nodes stored
    fn len(&self) -> usize;

    /// Detailed memory usage of this
    fn detailed_size(&self) -> String;

    /// Record that node id `nid` is in way id `wid`.
    fn insert(&mut self, nid: i64, wid: i64);

    /// Record that this nodes are in this way
    fn insert_many(&mut self, wid: i64, nids: &[i64]) {
        for nid in nids {
            self.insert(*nid, wid);
        }
    }

    /// True iff node id `nid` has been seen
    fn contains_nid(&self, nid: &i64) -> bool;

    /// Return all the ways that this node is in.
    fn ways<'a>(&'a self, nid: &i64) -> Box<dyn Iterator<Item = i64> + 'a>;
}

/// Some standard struct for doing this.
pub(crate) fn default() -> Box<dyn NodeIdWayIds> {
    Box::new(NodeIdWayIdsMultiMap::new())
}

/// Very simple BTreeMaps of nodeids:wayids
#[derive(Debug, GetSize)]
pub struct NodeIdWayIdsMultiMap {
    /// A node which is in exactly 1 way. Store the way id that it's in
    singles: BTreeMap<i64, i64>,

    /// Node which is in exactly 2 ways.
    doubles: BTreeMap<i64, (i64, i64)>,

    /// A node which is in many ways, Store the ways that it's in
    multiples: BTreeMap<i64, Vec<i64>>,
}

impl NodeIdWayIds for NodeIdWayIdsMultiMap {
    fn new() -> Self {
        NodeIdWayIdsMultiMap {
            singles: BTreeMap::new(),
            doubles: BTreeMap::new(),
            multiples: BTreeMap::new(),
        }
    }

    fn insert(&mut self, nid: i64, wid: i64) {
        if let Some(existing) = self.multiples.get_mut(&nid) {
            existing.push(wid);
            assert!(!self.singles.contains_key(&nid));
            assert!(!self.doubles.contains_key(&nid));
        } else if let Some((existing1, existing2)) = self.doubles.get(&nid) {
            if *existing1 != wid && *existing2 != wid {
                // upgrade to multiple
                self.multiples
                    .insert(nid, vec![*existing1, *existing2, wid]);
                self.doubles.remove(&nid);
                assert!(!self.singles.contains_key(&nid));
                assert!(!self.doubles.contains_key(&nid));
            } else {
                // already stored, do nothing
            }
        } else if let Some(existing) = self.singles.get(&nid) {
            if *existing != wid {
                // move to double
                assert!(!self.multiples.contains_key(&nid));
                assert!(!self.doubles.contains_key(&nid));
                self.doubles.insert(nid, (*existing, wid));
                self.singles.remove(&nid);
            } else {
                // do nothing
            }
        } else {
            self.singles.insert(nid, wid);
            assert!(!self.doubles.contains_key(&nid));
            assert!(!self.multiples.contains_key(&nid));
        }
    }

    fn contains_nid(&self, nid: &i64) -> bool {
        self.singles.contains_key(nid)
            || self.doubles.contains_key(nid)
            || self.multiples.contains_key(nid)
    }
    /// How many nodes have been saved
    fn len(&self) -> usize {
        self.singles.len() + self.doubles.len() + self.multiples.len()
    }

    fn ways<'a>(&'a self, nid: &i64) -> Box<dyn Iterator<Item = i64> + 'a> {
        if let Some(wid) = self.singles.get(nid) {
            Box::new(std::iter::once(*wid))
        } else if let Some((wid1, wid2)) = self.doubles.get(nid) {
            Box::new(std::iter::once(*wid1).chain(std::iter::once(*wid2)))
        } else if let Some(wids) = self.multiples.get(nid) {
            Box::new(wids.iter().map(|wid| *wid))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn detailed_size(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Size of nodeid_wayids: {} = {} bytes.\nnum_nodes: {} = {}.\nbytes/node={:>.2}\n",
            self.get_size(),
            self.get_size().to_formatted_string(&Locale::en),
            self.len(),
            self.len().to_formatted_string(&Locale::en),
            self.get_size() as f64 / self.len() as f64,
        ));
        output.push_str(&format!(
            "Size of nodeid_wayids.singles: {} = {} bytes, {} nodes\n",
            self.singles.get_size(),
            self.singles.get_size().to_formatted_string(&Locale::en),
            self.singles.len().to_formatted_string(&Locale::en)
        ));
        output.push_str(&format!(
            "Size of nodeid_wayids.doubles: {} = {} bytes, {} nodes\n",
            self.doubles.get_size(),
            self.doubles.get_size().to_formatted_string(&Locale::en),
            self.doubles.len().to_formatted_string(&Locale::en),
        ));
        output.push_str(&format!(
            "Size of nodeid_wayids.multiples: {} = {} bytes, {} nodes\n",
            self.multiples.get_size(),
            self.multiples.get_size().to_formatted_string(&Locale::en),
            self.multiples.len().to_formatted_string(&Locale::en),
        ));
        output
    }
}

const WAY_INDEX_BUCKET_SHIFT: i64 = 5;

/// More memory effecient, but slower.
#[derive(Debug, GetSize)]
pub struct NodeIdWayIdsWayIndex {
    /// For each wayid, a vartyint delta encoded sorted list of all the nodeids it's in.
    ways: BTreeMap<i32, Vec<u8>>,

    /// nodeid<<WAY_INDEX_BUCKET_SHIFT is the key
    /// value is unique list of wayids that have one of those nodeids
    nodeid_bucket_wayid: BTreeMap<i32, Vec<i32>>,

    /// Keep track of number of nodes
    num_nodes: usize,
}

impl NodeIdWayIdsWayIndex {
    fn set_nodeid_for_wayid(&mut self, wayid: i64, new_nodeid: i64) {
        let mut nodeids: Vec<i64> = self
            .get_nodeids_for_wayid(wayid)
            .unwrap_or_else(|| Vec::with_capacity(1));
        let old_num_nodes = nodeids.len();
        let wayid: i32 = wayid.try_into().expect("way id is too large for i32");
        nodeids.push(new_nodeid);
        nodeids.sort();
        nodeids.dedup();
        self.num_nodes = self.num_nodes + nodeids.len() - old_num_nodes;

        let new_nodeid_bucket = self.nodeid_bucket(new_nodeid);
        let bucket_wayids = self
            .nodeid_bucket_wayid
            .entry(new_nodeid_bucket)
            .or_default();
        bucket_wayids.push(wayid);
        bucket_wayids.sort();
        bucket_wayids.dedup();

        let nodeid_bytes = vartyint::write_many_delta_new(&nodeids);
        self.ways.insert(wayid, nodeid_bytes);
    }

    fn set_nodeids_for_wayid(&mut self, wayid: i64, new_nodeids: &[i64]) {
        let mut nodeids: Vec<i64> = self
            .get_nodeids_for_wayid(wayid)
            .unwrap_or_else(|| Vec::with_capacity(new_nodeids.len()));
        let old_num_nodes = nodeids.len();
        nodeids.extend(new_nodeids);
        nodeids.sort();
        nodeids.dedup();
        self.num_nodes = self.num_nodes + nodeids.len() - old_num_nodes;

        let wayid: i32 = wayid.try_into().expect("way id is too large for i32");
        let nodeid_bytes = vartyint::write_many_delta_new(&nodeids);
        self.ways.insert(wayid, nodeid_bytes);

        let mut new_nodeid_bucket;
        let mut bucket_wayids;
        for nid in new_nodeids {
            new_nodeid_bucket = self.nodeid_bucket(*nid);
            bucket_wayids = self
                .nodeid_bucket_wayid
                .entry(new_nodeid_bucket)
                .or_default();
            bucket_wayids.push(wayid);
            bucket_wayids.sort();
            bucket_wayids.dedup();
        }
    }

    fn get_nodeids_for_wayid(&self, wayid: impl Into<i64>) -> Option<Vec<i64>> {
        let wayid: i64 = wayid.into();
        let wayid: i32 = wayid.try_into().expect("way id is too large for i32. This tool uses optimizations which assume wayid < 2³²");
        match self.ways.get(&wayid) {
            None => None,
            Some(nodeid_bytes) => {
                let nodeids: Vec<i64> = vartyint::read_many_delta_new(nodeid_bytes).unwrap();
                Some(nodeids)
            }
        }
    }
    fn get_nodeids_for_wayid_iter(&self, wayid: impl Into<i64>) -> impl Iterator<Item = i64> + '_ {
        let wayid: i64 = wayid.into();
        let wayid: i32 = wayid.try_into().expect("way id is too large for i32. This tool uses optimizations which assume wayid < 2³²");
        self.ways
            .get(&wayid)
            .into_iter()
            .flat_map(|nodeid_bytes| vartyint::read_many_delta(nodeid_bytes).map(|r| r.unwrap()))
    }

    fn nodeid_bucket(&self, nid: i64) -> i32 {
        let bucket: i32 = (nid >> WAY_INDEX_BUCKET_SHIFT)
            .try_into()
            .expect("Node id >> by WAY_INDEX_BUCKET_SHIFT is too large to fit in i32. This tool uses optimizations which assume wayid < 2³²");
        bucket
    }
}

impl NodeIdWayIds for NodeIdWayIdsWayIndex {
    fn new() -> Self
    where
        Self: Sized + Send,
    {
        NodeIdWayIdsWayIndex {
            ways: BTreeMap::new(),
            num_nodes: 0,
            nodeid_bucket_wayid: BTreeMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.num_nodes
    }
    fn detailed_size(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Size of nodeid_wayids: {} = {} bytes.\nnum_nodes: {} = {}.\nbytes/node={:>.2}\n",
            self.get_size(),
            self.get_size().to_formatted_string(&Locale::en),
            self.len(),
            self.len().to_formatted_string(&Locale::en),
            self.get_size() as f64 / self.len() as f64,
        ));
        output.push_str(&format!(
            "Size of self.ways: {} = {} bytes num_ways: {} = {} \n",
            self.ways.get_size(),
            self.ways.get_size().to_formatted_string(&Locale::en),
            self.ways.len(),
            self.ways.len().to_formatted_string(&Locale::en),
        ));
        output.push_str(&format!(
            "Size of self.nodeid_bucket_wayid: {} = {} bytes num_ways: {} = {} \n",
            self.nodeid_bucket_wayid.get_size(),
            self.nodeid_bucket_wayid
                .get_size()
                .to_formatted_string(&Locale::en),
            self.nodeid_bucket_wayid.len(),
            self.nodeid_bucket_wayid
                .len()
                .to_formatted_string(&Locale::en),
        ));
        output
    }

    /// Record that node id `nid` is in way id `wid`.
    fn insert(&mut self, nid: i64, wid: i64) {
        self.set_nodeid_for_wayid(wid, nid);
    }

    /// Record that this nodes are in this way
    fn insert_many(&mut self, wid: i64, nids: &[i64]) {
        self.set_nodeids_for_wayid(wid, nids);
    }

    /// True iff node id `nid` has been seen
    fn contains_nid(&self, nid: &i64) -> bool {
        let new_nodeid_bucket = self.nodeid_bucket(*nid);
        match self.nodeid_bucket_wayid.get(&new_nodeid_bucket) {
            None => false,
            Some(wids) => wids.iter().any(|wid| {
                self.get_nodeids_for_wayid(*wid)
                    .map_or(false, |nids| nids.contains(nid))
            }),
        }
    }

    /// Return all the ways that this node is in.
    fn ways<'a>(&'a self, nid: &i64) -> Box<dyn Iterator<Item = i64> + 'a> {
        let nid = *nid;
        let bucketid = self.nodeid_bucket(nid);
        Box::new(
            self.nodeid_bucket_wayid
                .get(&bucketid)
                .into_iter()
                .flat_map(|wids| wids.into_iter())
                .filter(move |wid| {
                    self.get_nodeids_for_wayid_iter(**wid)
                        .any(|this_nid| this_nid == nid)
                })
                .map(|wid| (*wid).into()),
        )
    }
}
