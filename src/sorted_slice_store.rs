use rayon::prelude::*;
use std::borrow::Borrow;
use std::collections::HashMap;

#[derive(Debug)]
pub struct SortedSliceMap<K, V> {
    data: Box<[(K, V)]>,
}

impl<K, V> SortedSliceMap<K, V>
where
    K: Ord + Send,
    V: Send,
{
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter(src: impl Iterator<Item = (K, V)>) -> Self {
        let data: Vec<(K, V)> = src.collect();
        Self::from_vec(data)
    }

    pub fn from_w_size(src: impl ExactSizeIterator<Item = (K, V)>) -> Self {
        let mut data = Vec::with_capacity(src.len());
        data.extend(src);
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
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &(K, V)> {
        self.data.iter()
    }
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut (K, V)> {
        self.data.iter_mut()
    }
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> {
        self.data.iter().map(|(k, _)| k)
    }
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &(K, V)>
    where
        K: Sync,
        V: Sync,
    {
        self.data.as_ref().par_iter()
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.data.binary_search_by_key(&k, |(k2, _v)| k2).is_ok()
    }

    pub fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + Ord + ?Sized,
    {
        self.data
            .binary_search_by_key(&k, |(k2, _v)| k2.borrow())
            .ok()
            .map(|i| &self.data[i].1)
    }

    pub fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq + Ord + ?Sized,
    {
        self.data
            .binary_search_by_key(&k, |(k2, _v)| k2.borrow())
            .ok()
            .and_then(|i| self.data.get_mut(i))
            .map(|(_k, v)| v)
    }

    pub fn set(&mut self, k: &K, new_value: V) {
        let idx = self.data.binary_search_by_key(&k, |(k2, _v)| k2).unwrap();
        self.data[idx].1 = new_value;
    }
}

impl<K, V> From<std::collections::HashMap<K, V>> for SortedSliceMap<K, V>
where
    K: Ord + Send,
    V: Send,
{
    fn from(orig: HashMap<K, V>) -> Self {
        SortedSliceMap::from_w_size(orig.into_iter())
    }
}

impl<K, V> Clone for SortedSliceMap<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

#[derive(Debug)]
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
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter(src: impl Iterator<Item = T>) -> Self {
        let data: Vec<T> = src.collect();
        Self::from_vec(data)
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn contains(&self, value: &T) -> bool {
        self.data.binary_search(value).is_ok()
    }

    pub fn idx(&self, value: &T) -> Option<usize> {
        self.data.binary_search(value).ok()
    }
    pub fn get_by_idx(&self, idx: impl TryInto<usize>) -> Option<&T> {
        match idx.try_into() {
            Ok(idx) => self.data.get(idx),
            _ => None,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &T>
    where
        T: Sync,
    {
        self.data.as_ref().par_iter()
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
