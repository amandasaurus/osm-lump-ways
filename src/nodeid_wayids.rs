//! A struct for storing which wayid(s) a nodeid is in.
//! Most nodes are in only 1 way. This struct uses much less memory by taking advantage of that.
//! A small amount of nodes are in exactly 2 nodes (This saves about 10% space)
use super::*;
use std::collections::BTreeMap;
use std::fmt::Debug;

pub(crate) trait NodeIdWayIds: Debug + Send + Sync {
    fn new() -> Self
    where
        Self: Sized + Send;

    /// Number of nodes stored
    fn len(&self) -> usize;
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
    fn ways<'a>(&'a self, nid: &i64) -> Box<dyn Iterator<Item = &i64> + 'a>;
}

/// Some standard struct for doing this.
pub(crate) fn default() -> Box<dyn NodeIdWayIds> {
    Box::new(NodeIdWayIdsMultiMap::new())
}

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

    fn ways<'a>(&'a self, nid: &i64) -> Box<dyn Iterator<Item = &i64> + 'a> {
        if let Some(wid) = self.singles.get(nid) {
            Box::new(std::iter::once(wid))
        } else if let Some((wid1, wid2)) = self.doubles.get(nid) {
            Box::new(std::iter::once(wid1).chain(std::iter::once(wid2)))
        } else if let Some(wids) = self.multiples.get(nid) {
            Box::new(wids.iter())
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn detailed_size(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Size of nodeid_wayids: {} = {} bytes num_nodes: {} = {}\n",
            self.get_size(),
            self.get_size().to_formatted_string(&Locale::en),
            self.len(),
            self.len().to_formatted_string(&Locale::en)
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
