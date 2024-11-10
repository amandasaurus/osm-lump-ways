use rayon::prelude::*;
use std::borrow::Borrow;

pub struct SortedSliceMap<K, V> {
    data: Box<[(K, V)]>,
}

impl<K, V> SortedSliceMap<K, V>
where
    K: Ord + Send,
    V: Send,
{
    pub fn from(src: impl Iterator<Item = (K, V)>) -> Self {
        let data: Vec<(K, V)> = src.collect();
        Self::from_vec(data)
    }

    pub fn from_vec(mut data: Vec<(K, V)>) -> Self {
        data.par_sort_unstable_by(|(k1, _v1), (k2, _v2)| k1.cmp(k2));
        data.dedup_by(|(k1, _v), (k2, _v2)| k1 == k2);

        SortedSliceMap {
            data: data.into_boxed_slice(),
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.data.iter()
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.data.binary_search_by_key(&k, |(k2, _v)| k2).is_ok()
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.data
            .binary_search_by_key(&k, |(k2, _v)| k2)
            .ok()
            .map(|i| &self.data[i].1)
    }
}

pub struct SortedSliceSet<T> {
    data: Box<[T]>,
}

impl<T: Ord + Send> SortedSliceSet<T> {
    pub fn from_vec(mut data: Vec<T>) -> Self {
        data.par_sort_unstable();
        data.dedup();

        SortedSliceSet {
            data: data.into_boxed_slice(),
        }
    }
    pub fn from(src: impl Iterator<Item = T>) -> Self {
        let mut data: Vec<T> = src.collect();
        Self::from_vec(data)
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn contains(&self, value: &T) -> bool {
        self.data.binary_search(value).is_ok()
    }
}

impl<T> From<Vec<T>> for SortedSliceSet<T>
where
    T: Ord + Send,
{
    fn from(v: Vec<T>) -> Self {
        SortedSliceSet::from_vec(v)
    }
}
