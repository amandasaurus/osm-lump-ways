//! A struct for storing which wayid(s) a nodeid is in.
//! Most nodes are in only 1 way. This struct uses much less memory by taking advantage of that.
use super::*;

#[derive(Debug, GetSize)]
pub struct NodeIdWayIds {
    /// A node which is in exactly 1 way. Store the way id that it's in
    singles: HashMap<i64, i64>,

    /// A node which is in many ways, Store the ways that it's in
    multiples: HashMap<i64, Vec<i64>>,
}

impl NodeIdWayIds {
    pub fn new() -> Self {
        NodeIdWayIds {
            singles: HashMap::new(),
            multiples: HashMap::new(),
        }
    }

    /// Record that node id `nid` is in way id `wid`.
    pub fn insert(&mut self, nid: i64, wid: i64) {
        if let Some(existing) = self.multiples.get_mut(&nid) {
            existing.push(wid);
            assert!(!self.singles.contains_key(&nid));
        } else if let Some(existing) = self.singles.get(&nid) {
            if *existing != wid {
                // move to multiple
                self.multiples.insert(nid, vec![*existing, wid]);
                self.singles.remove(&nid);
            } else {
                // do nothing
            }
        } else {
            self.singles.insert(nid, wid);
        }
    }

    /// True iff node id `nid` has been seen
    pub fn contains_nid(&self, nid: &i64) -> bool {
        self.singles.contains_key(nid) || self.multiples.contains_key(nid)
    }
    /// How many nodes have been saved
    pub fn len(&self) -> usize {
        self.singles.len() + self.multiples.len()
    }

    /// Return all the ways that this node is in.
    pub fn ways<'a>(&'a self, nid: &i64) -> Box<dyn Iterator<Item = &i64> + 'a> {
        if let Some(wid) = self.singles.get(nid) {
            Box::new(std::iter::once(wid))
        } else if let Some(wids) = self.multiples.get(nid) {
            Box::new(wids.iter())
        } else {
            Box::new(std::iter::empty())
        }
    }
}
