//! This is a BTreeMap<i64, V>, but it stores it in several levels of BTreeMap, by splitting the
//! i64 key.
//! Goal: Reduce memory usage of struct, by storing less of the key.
//! End result: ~5% memory reduction of total programme. Not very impressive.
use get_size::GetSize;
use rayon::prelude::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, GetSize)]
pub struct BTreeMapSplitKey<V> {
    inner: BTreeMap<i32, BTreeMap<i32, V>>,
}

const I32_LIMIT: i64 = i32::MAX as i64;

fn split_key(k: &i64) -> [i32; 2] {
    [
        (k / I32_LIMIT).try_into().expect("Bad presumption for i32"),
        (k % I32_LIMIT) as i32,
    ]
}

fn join_key(ks: &[i32; 2]) -> i64 {
    (ks[0] as i64 * I32_LIMIT) + (ks[1] as i64)
}

#[allow(dead_code)]
impl<V> BTreeMapSplitKey<V>
where
    V: Sync,
{
    pub fn new() -> Self {
        BTreeMapSplitKey {
            inner: BTreeMap::new(),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (i64, &V)> {
        self.inner
            .iter()
            .flat_map(|(k0, i2)| i2.iter().map(|(k1, v)| (join_key(&[*k0, *k1]), v)))
    }
    pub fn into_iter(self) -> impl Iterator<Item = (i64, V)> {
        self.inner
            .into_iter()
            .flat_map(|(k0, i2)| i2.into_iter().map(move |(k1, v)| (join_key(&[k0, k1]), v)))
    }
    pub fn par_iter(&self) -> impl ParallelIterator<Item = (i64, &V)> {
        self.inner
            .par_iter()
            .flat_map(|(k0, i2)| i2.par_iter().map(|(k1, v)| (join_key(&[*k0, *k1]), v)))
    }

    pub fn get(&self, key: &i64) -> Option<&V> {
        let k = split_key(key);
        self.inner.get(&k[0]).and_then(|i2| i2.get(&k[1]))
    }
    pub fn get_mut(&mut self, key: &i64) -> Option<&mut V> {
        let k = split_key(key);
        self.inner.get_mut(&k[0]).and_then(|i2| i2.get_mut(&k[1]))
    }
    pub fn keys(&self) -> impl Iterator<Item = i64> + '_ {
        self.inner
            .iter()
            .flat_map(|(k0, i2)| i2.keys().map(|k1| join_key(&[*k0, *k1])))
    }
    pub fn contains_key(&self, key: &i64) -> bool {
        let k = split_key(key);
        self.inner
            .get(&k[0])
            .map_or(false, |i2| i2.contains_key(&k[1]))
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn remove(&mut self, key: &i64) -> Option<V> {
        let k = split_key(key);
        if let Some(i2) = self.inner.get_mut(&k[0]) {
            let res = i2.remove(&k[1]);
            if i2.is_empty() {
                self.inner.remove(&k[0]);
            }
            res
        } else {
            None
        }
    }
    pub fn entry(&mut self, key: i64) -> std::collections::btree_map::Entry<i32, V> {
        let k = split_key(&key);
        self.inner.entry(k[0]).or_default().entry(k[1])
    }
    pub fn len(&self) -> usize {
        self.inner.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split1() {
        assert_eq!(split_key(&0), [0, 0]);
    }
    #[test]
    fn split2() {
        assert_eq!(split_key(&I32_LIMIT), [1, 0]);
    }
    #[test]
    fn split3() {
        assert_eq!(split_key(&(I32_LIMIT + 1)), [1, 1]);
    }
    #[test]
    fn split4() {
        assert_eq!(split_key(&(I32_LIMIT + 2)), [1, 2]);
    }
    #[test]
    fn split5() {
        assert_eq!(split_key(&(I32_LIMIT - 1)), [0, 2147483646]);
    }
    #[test]
    fn split6() {
        assert_eq!(split_key(&2147483646), [0, 2147483646]);
    }
    #[test]
    fn split7() {
        assert_eq!(split_key(&(2147483646 + 1)), [1, 0]);
    }

    #[test]
    fn join1() {
        assert_eq!(join_key(&[0, 0]), 0);
    }
    #[test]
    fn join2() {
        assert_eq!(join_key(&[1, 0]), 2147483647);
    }
    #[test]
    fn join3() {
        assert_eq!(join_key(&[1, 1]), 2147483648);
    }
    #[test]
    fn join4() {
        assert_eq!(join_key(&[1, 2]), 2147483649);
    }
    #[test]
    fn join5() {
        assert_eq!(join_key(&[0, 2147483646]), 2147483646);
    }

    #[test]
    fn simple1() {
        let mut x = BTreeMapSplitKey::new();
        *x.entry(0).or_default() = 12;
        assert_eq!(x.get(&0), Some(&12));
        assert_eq!(x.get(&1), None);
        assert_eq!(x.iter().collect::<Vec<_>>(), vec![(0, &12)])
    }

    #[test]
    fn empty1() {
        let mut x = BTreeMapSplitKey::new();
        assert!(x.is_empty());
        assert_eq!(x.len(), 0);
        *x.entry(0).or_default() = 12;
        assert_eq!(x.len(), 1);
        assert!(!x.is_empty());
        x.remove(&0);
        assert!(x.is_empty());
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn len1() {
        let mut x = BTreeMapSplitKey::new();
        assert_eq!(x.len(), 0);
        *x.entry(0).or_default() = 12_usize;
        assert_eq!(x.len(), 1);
        *x.entry(1).or_default() = 12_usize;
        assert_eq!(x.len(), 2);
        *x.entry(1).or_default() = 12_usize;
        assert_eq!(x.len(), 2);

        *x.entry(I32_LIMIT + 2).or_default() = 13_usize;

        assert_eq!(x.len(), 3);
    }
}
