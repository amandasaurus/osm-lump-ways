//! A struct for storing which wayid(s) a nodeid is in.
//! Most nodes are in only 1 way. This struct uses much less memory by taking advantage of that.
//! A small amount of nodes are in exactly 2 nodes (This saves about 10% space)
use super::*;
use std::collections::BTreeMap;
use std::fmt::Debug;

/// Something which stores which nodeids are in which wayid
pub trait NodeIdWayIds: Debug + Send + Sync {
    /// Create new version
    fn new() -> Self
    where
        Self: Sized + Send;

    /// Number of nodes stored
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

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
    fn ways(&self, nid: &i64) -> Box<dyn Iterator<Item = i64> + '_>;

    // returns true iff this node id is in >1 way
    fn nid_is_in_many(&self, nid: &i64) -> bool;
}

/// Some standard struct for doing this.
pub fn default() -> impl NodeIdWayIds {
    //Box::new(NodeIdWayIdsMultiMap::new())
    NodeIdWayIdsAuto::new()
}

/// Very simple BTreeMaps of nodeids:wayids
#[derive(Debug, GetSize, Default)]
pub struct NodeIdWayIdsMultiMap {
    /// A node which is in exactly 1 way. Store the way id that it's in
    singles: BTreeMap<i64, i64>,

    /// A node which is in many ways, Store the ways that it's in
    multiples: BTreeMap<i64, Vec<i64>>,
}

impl NodeIdWayIdsMultiMap {
    fn drain_all(self) -> impl Iterator<Item = (i64, i64)> {
        let NodeIdWayIdsMultiMap { singles, multiples } = self;

        Box::new(
            singles.into_iter().chain(
                multiples
                    .into_iter()
                    .flat_map(|(nid, wids)| wids.into_iter().map(move |wid| (nid, wid))),
            ),
        )
    }
}

impl NodeIdWayIds for NodeIdWayIdsMultiMap {
    fn new() -> Self {
        NodeIdWayIdsMultiMap {
            singles: BTreeMap::new(),
            multiples: BTreeMap::new(),
        }
    }

    fn insert(&mut self, nid: i64, wid: i64) {
        if let Some(existing) = self.multiples.get_mut(&nid) {
            existing.push(wid);
            assert!(!self.singles.contains_key(&nid));
        } else if let Some(existing) = self.singles.get(&nid) {
            if *existing != wid {
                // move to multiple
                assert!(!self.multiples.contains_key(&nid));
                self.multiples.insert(nid, vec![*existing, wid]);
                self.singles.remove(&nid);
            } else {
                // same value
                // do nothing
            }
        } else {
            self.singles.insert(nid, wid);
            assert!(!self.multiples.contains_key(&nid));
        }
    }

    fn contains_nid(&self, nid: &i64) -> bool {
        self.singles.contains_key(nid) || self.multiples.contains_key(nid)
    }
    /// How many nodes have been saved
    fn len(&self) -> usize {
        self.singles.len() + self.multiples.len()
    }

    fn nid_is_in_many(&self, nid: &i64) -> bool {
        self.multiples.contains_key(nid)
    }

    fn ways(&self, nid: &i64) -> Box<dyn Iterator<Item = i64> + '_> {
        if let Some(wid) = self.singles.get(nid) {
            Box::new(std::iter::once(*wid))
        } else if let Some(wids) = self.multiples.get(nid) {
            Box::new(wids.iter().copied())
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
            "Size of nodeid_wayids.multiples: {} = {} bytes, {} nodes\n",
            self.multiples.get_size(),
            self.multiples.get_size().to_formatted_string(&Locale::en),
            self.multiples.len().to_formatted_string(&Locale::en),
        ));
        output
    }
}

/// More memory effecient, but slower.
#[derive(Debug, GetSize)]
pub struct NodeIdWayIdsBucketWayIndex {
    /// For each wayid, a vartyint delta encoded sorted list of all the nodeids it's in.
    ways: BTreeMap<i32, Vec<u8>>,

    /// nodeid<<WAY_INDEX_BUCKET_SHIFT is the key
    /// value is unique list of wayids that have one of those nodeids
    nodeid_bucket_wayid: BTreeMap<i32, Vec<i32>>,

    /// Keep track of number of nodes
    num_nodes: usize,

    bucket_shift: i64,
}

impl NodeIdWayIdsBucketWayIndex {
    fn with_bucket(bucket_shift: i64) -> Self {
        Self {
            ways: BTreeMap::new(),
            num_nodes: 0,
            nodeid_bucket_wayid: BTreeMap::new(),
            bucket_shift,
        }
    }

    fn bucket_shift(&self) -> i64 {
        self.bucket_shift
    }

    fn set_nodeid_for_wayid(&mut self, wayid: i64, new_nodeid: i64) {
        let mut nodeids: Vec<i64> = self.get_nodeids_for_wayid(wayid);
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
        let mut nodeids: Vec<i64> = self.get_nodeids_for_wayid(wayid);
        nodeids.reserve(new_nodeids.len());
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

    fn get_nodeids_for_wayid(&self, wayid: impl Into<i64>) -> Vec<i64> {
        self.get_nodeids_for_wayid_iter(wayid).collect::<Vec<_>>()
    }
    fn get_nodeids_for_wayid_iter(&self, wayid: impl Into<i64>) -> impl Iterator<Item = i64> + '_ {
        let wayid: i64 = wayid.into();
        let wayid: i32 = wayid.try_into().expect(
            "way id is too large for i32. This tool uses optimizations which assume wayid < 2³²",
        );
        self.ways
            .get(&wayid)
            .into_iter()
            .flat_map(|nodeid_bytes| vartyint::read_many_delta(nodeid_bytes).map(|r| r.unwrap()))
    }

    fn nodeid_bucket(&self, nid: i64) -> i32 {
        let bucket: i32 = (nid >> self.bucket_shift())
            .try_into()
            .expect("Node id >> by WAY_INDEX_BUCKET_SHIFT is too large to fit in i32. This tool uses optimizations which assume wayid < 2³²");
        bucket
    }

    fn ways_for_nid(&self, nid: &i64) -> impl Iterator<Item = &i32> {
        let nid = *nid;
        let bucketid = self.nodeid_bucket(nid);
        self.nodeid_bucket_wayid
            .get(&bucketid)
            .into_iter()
            .flat_map(|wids| wids.iter())
            .filter(move |wid| {
                self.get_nodeids_for_wayid_iter(**wid)
                    .take_while(|this_nid| *this_nid <= nid) // stored in order, so early exit
                    // possible
                    .any(|this_nid| this_nid == nid)
            })
    }
}

impl NodeIdWayIds for NodeIdWayIdsBucketWayIndex {
    fn new() -> Self {
        Self::with_bucket(6)
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
                self.get_nodeids_for_wayid_iter(*wid)
                    .take_while(|this_nid| this_nid <= nid) // stored in order, so early exit
                    // possible
                    .any(|w_nid| w_nid == *nid)
            }),
        }
    }

    fn nid_is_in_many(&self, nid: &i64) -> bool {
        // Ask for the ways this nid is in, and check there are >1
        self.ways_for_nid(nid).nth(1).is_some()
    }

    /// Return all the ways that this node is in.
    fn ways(&self, nid: &i64) -> Box<dyn Iterator<Item = i64> + '_> {
        Box::new(self.ways_for_nid(nid).map(|wid32| (*wid32).into()))
    }
}

#[derive(Debug)]
enum NodeIdWayIdsAuto {
    MultiMap(NodeIdWayIdsMultiMap),
    BucketMap(NodeIdWayIdsBucketWayIndex),
}

impl NodeIdWayIdsAuto {
    fn possibly_switch_backend(&mut self) {
        if let Self::MultiMap(ref mut multi_map) = self {
            if multi_map.len() > SWITCH_TO_BUCKET {
                let multi_map = std::mem::take(multi_map);
                let started_conversion = std::time::Instant::now();
                info!("There are {} nodes in the nodeid:wayid (> {}). Switching from CPU-faster memory-ineffecient MultiMap, to CPU-slower memory-effecientier Bucket Index", multi_map.len().to_formatted_string(&Locale::en), SWITCH_TO_BUCKET);
                debug!("Old object: {}", multi_map.detailed_size());
                let old_size = multi_map.get_size();

                // Create a new bucket and convert the old to this.
                let mut new_bucket = NodeIdWayIdsBucketWayIndex::with_bucket(7);
                for (nid, wid) in multi_map.drain_all() {
                    new_bucket.insert(nid, wid);
                }

                let converstion_duration = std::time::Instant::now() - started_conversion;
                debug!("New object: {}", new_bucket.detailed_size());
                debug!(
                    "It took {} sec to convert to bucket index",
                    converstion_duration.as_secs()
                );
                debug!(
                    "New index is {}% the size of the old one",
                    (100 * new_bucket.get_size()) / old_size
                );

                // and we're that now
                *self = Self::BucketMap(new_bucket);
            }
        }
    }
}

/// After this many nodes, switch to the CPU slower, but RAM-smaller Bucket Way Index
const SWITCH_TO_BUCKET: usize = 100_000_000;

impl NodeIdWayIds for NodeIdWayIdsAuto {
    fn new() -> Self {
        NodeIdWayIdsAuto::MultiMap(NodeIdWayIdsMultiMap::new())
    }

    /// Number of nodes stored
    fn len(&self) -> usize {
        match self {
            Self::MultiMap(x) => x.len(),
            Self::BucketMap(x) => x.len(),
        }
    }

    /// Detailed memory usage of this
    fn detailed_size(&self) -> String {
        match self {
            Self::MultiMap(x) => x.detailed_size(),
            Self::BucketMap(x) => x.detailed_size(),
        }
    }

    /// Record that node id `nid` is in way id `wid`.
    fn insert(&mut self, nid: i64, wid: i64) {
        self.possibly_switch_backend();
        match self {
            Self::MultiMap(x) => x.insert(nid, wid),
            Self::BucketMap(x) => x.insert(nid, wid),
        }
    }

    fn insert_many(&mut self, wid: i64, nids: &[i64]) {
        self.possibly_switch_backend();
        match self {
            Self::MultiMap(x) => x.insert_many(wid, nids),
            Self::BucketMap(x) => x.insert_many(wid, nids),
        }
    }

    /// True iff node id `nid` has been seen
    fn contains_nid(&self, nid: &i64) -> bool {
        match self {
            Self::MultiMap(x) => x.contains_nid(nid),
            Self::BucketMap(x) => x.contains_nid(nid),
        }
    }

    fn nid_is_in_many(&self, nid: &i64) -> bool {
        match self {
            Self::MultiMap(x) => x.nid_is_in_many(nid),
            Self::BucketMap(x) => x.nid_is_in_many(nid),
        }
    }

    /// Return all the ways that this node is in.
    fn ways(&self, nid: &i64) -> Box<dyn Iterator<Item = i64> + '_> {
        match self {
            Self::MultiMap(x) => x.ways(nid),
            Self::BucketMap(x) => x.ways(nid),
        }
    }
}
