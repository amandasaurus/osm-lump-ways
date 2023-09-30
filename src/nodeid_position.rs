use super::*;
use std::collections::{HashMap, BTreeMap};
use osmio::{Lat, Lon};

pub trait NodeIdPosition: std::fmt::Debug + std::marker::Send+std::marker::Sync {
    fn new() -> Self where Self: Sized;

    fn insert(&mut self, node_id: i64, pos: (f64, f64));

    fn contains_key(&self, node_id: &i64) -> bool;
    fn get(&self, node_id: &i64) -> Option<(f64, f64)>;

    fn len(&self) -> usize;

    fn retain_by_key(&mut self, f: impl FnMut(&i64) -> bool);

    fn extend<I: IntoIterator<Item=(i64, (f64, f64))>>(&mut self, iter: I) {
        for el in iter {
            self.insert(el.0, el.1);
        }
    }

    fn detailed_size(&self) -> String;

    fn shrink_to_fit(&mut self) {}

}

/// A default good value
pub(crate) fn default() -> impl NodeIdPosition {
    NodeIdPositionMap::new()
}

#[derive(Debug, GetSize)]
pub struct NodeIdPositionMap {
    inner: BTreeMap<i64, (i32, i32)>,
}


impl NodeIdPosition for NodeIdPositionMap {
    fn new() -> Self {
        NodeIdPositionMap{ inner: BTreeMap::new() }
    }

    fn insert(&mut self, node_id: i64, pos: (f64, f64)) {
        let pos = (Lat::try_from(pos.0).unwrap().inner(), Lon::try_from(pos.1).unwrap().inner());
        self.inner.insert(node_id, pos);
    }

    fn contains_key(&self, node_id: &i64) -> bool {
        self.inner.contains_key(node_id)
    }

    fn get(&self, node_id: &i64) -> Option<(f64, f64)> {
        self.inner.get(node_id).map(|(lat, lng)| (Lat::from_inner(*lat).degrees(), Lon::from_inner(*lng).degrees()))
     }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn retain_by_key(&mut self, mut f: impl FnMut(&i64) -> bool )
    {
        self.inner.retain(|k, _v| f(k));
    }

    fn detailed_size(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Size of nodeid:pos: {} = {} bytes\n", self.get_size(), self.get_size().to_formatted_string(&Locale::en)));
        output
    }

    fn extend<I: IntoIterator<Item=(i64, (f64, f64))>>(&mut self, iter: I) {
        for el in iter {
            self.insert(el.0, ((el.1.0.try_into().unwrap(), el.1.1.try_into().unwrap())));
        }
    }

}
