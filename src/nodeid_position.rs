use super::*;
use std::collections::{HashMap, BTreeMap};

#[derive(Debug, GetSize)]
pub struct NodeIdPosition {
    inner: BTreeMap<i64, (f64, f64)>,
}

impl NodeIdPosition {
    pub fn new() -> Self {
        NodeIdPosition{ inner: BTreeMap::new() }
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new()
    }

    pub fn insert(&mut self, node_id: i64, pos: (f64, f64)) {
        self.inner.insert(node_id, pos);
    }

    pub fn contains_key(&self, node_id: &i64) -> bool {
        self.inner.contains_key(node_id)
    }
    pub fn get(&self, node_id: &i64) -> Option<&(f64, f64)> {
        self.inner.get(node_id)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn retain_by_key(&mut self, mut f: impl FnMut(&i64) -> bool )
    {
        self.inner.retain(|k, _v| f(k));
    }

    // Not available for BTreeMap
    pub fn shrink_to_fit(&mut self) {}
    pub fn reserve(&mut self, addition: usize) {}

    pub fn detailed_size(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Size of nodeid:pos: {} = {} bytes\n", self.get_size(), self.get_size().to_formatted_string(&Locale::en)));
        output
    }


}

impl FromIterator<(i64, (f64, f64))> for NodeIdPosition {
    fn from_iter<I: IntoIterator<Item=(i64, (f64, f64))>>(iter: I) -> Self {
        let mut np = NodeIdPosition::new();
        np.extend(iter);
        np
    }
}

impl Extend<(i64, (f64, f64))> for NodeIdPosition {
    fn extend<I: IntoIterator<Item=(i64, (f64, f64))>>(&mut self, iter: I) {
        for el in iter {
            self.insert(el.0, ((el.1.0.try_into().unwrap(), el.1.1.try_into().unwrap())));
        }
    }
}
