/// Storing the position of nodes based on their node id
use super::*;
use osmio::{Lat, Lon};
use std::collections::BTreeMap;

#[allow(dead_code)]
/// Store the position of a node based on it's id
pub trait NodeIdPosition: std::fmt::Debug + std::marker::Send + std::marker::Sync {
    fn new() -> Self
    where
        Self: Sized;

    /// Set this position
    fn insert(&mut self, node_id: i64, pos: (f64, f64));

    /// Set this position (f64 explict)
    fn insert_f64(&mut self, node_id: i64, pos: (f64, f64)) {
        self.insert(node_id, pos);
    }
    /// Set this position if already have explicit inner format
    fn insert_i32(&mut self, node_id: i64, pos: (i32, i32)) {
        let pos: (f64, f64) = (
            Lat::from_inner(pos.0).degrees(),
            Lon::from_inner(pos.1).degrees(),
        );
        self.insert_f64(node_id, pos);
    }

    fn contains_key(&self, node_id: &i64) -> bool {
        self.get(node_id).is_ok()
    }
    /// Return the location for this node id
    fn get(&self, node_id: &i64) -> Result<(f64, f64)>;

    fn get_many_unwrap(&self, nids: &[i64], output: &mut [(f64, f64)]) {
        for (nid, output) in nids.iter().zip(output.iter_mut()) {
            *output = self.get(nid).unwrap();
        }
    }

    fn get_many(&self, nids: &[i64], output: &mut [Result<(f64, f64)>]) {
        for i in 0..nids.len() {
            output[i] = self.get(&nids[i]);
        }
    }

    /// Number of nodes inside
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn detailed_size(&self) -> String;

    fn shrink_to_fit(&mut self) {}

    fn finished_inserting(&mut self) {}
}

/// A default good value
pub fn default() -> impl NodeIdPosition {
    //NodeIdPositionMap::new()
    NodeIdPositionBucket::with_bucket(3)
}

/// A simple map
// IME BTreeMap is smaller than HashMap. Positions are stored as i32
#[derive(Debug, GetSize)]
pub struct NodeIdPositionMap {
    inner: BTreeMap<i64, (i32, i32)>,
}

impl NodeIdPosition for NodeIdPositionMap {
    fn new() -> Self {
        NodeIdPositionMap {
            inner: BTreeMap::new(),
        }
    }

    fn insert_i32(&mut self, node_id: i64, pos: (i32, i32)) {
        self.inner.insert(node_id, pos);
    }

    fn insert(&mut self, node_id: i64, pos: (f64, f64)) {
        let pos = (
            Lat::try_from(pos.0).unwrap().inner(),
            Lon::try_from(pos.1).unwrap().inner(),
        );
        self.insert_i32(node_id, pos);
    }

    fn contains_key(&self, node_id: &i64) -> bool {
        self.inner.contains_key(node_id)
    }

    fn get(&self, node_id: &i64) -> Result<(f64, f64)> {
        self.inner
            .get(node_id)
            .map(|(lat, lng)| {
                (
                    Lat::from_inner(*lat).degrees(),
                    Lon::from_inner(*lng).degrees(),
                )
            })
            .ok_or_else(|| anyhow::anyhow!("Couldn't get node position for nid {}", node_id))
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn detailed_size(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Size of nodeid_pos (NodeIdPositionMap): {} = {} bytes.\nnum_nodes: {} = {}.\nbytes/node = {:>.2}\n",
            self.get_size(),
            self.get_size().to_formatted_string(&Locale::en),
            self.len(),
            self.len().to_formatted_string(&Locale::en),
            self.get_size() as f64 / self.len() as f64,
        ));
        output
    }
}

/// A memory effecient node location store
/// nodes with the same id are often very close together. This will bucket nodes based on id, and
/// store several relative offsets (in a compressed form)
#[derive(Debug, GetSize)]
pub struct NodeIdPositionBucket {
    /// left shift node ids by this much. Max 6
    bucket_shift: i64,
    num_nodes: usize,

    /// All the data is here. Key is the bucket id
    inner: BTreeMap<i32, Vec<u8>>,

    /// A local cache of decoded values to make lookup & inserts faster
    #[allow(clippy::type_complexity)]
    cache: Option<(i32, Vec<Option<(i32, i32)>>)>,
}

impl NodeIdPositionBucket {
    /// Create a new object with this shift
    fn with_bucket(bucket_shift: i64) -> Self {
        NodeIdPositionBucket {
            bucket_shift,
            num_nodes: 0,
            inner: BTreeMap::new(),
            cache: None,
        }
    }

    /// a simple getter
    fn bucket_shift(&self) -> i64 {
        self.bucket_shift
    }

    /// Return the bucket id, and the local offset within that bucket for this nodeid
    fn nodeid_bucket_local(&self, nid: i64) -> (i32, usize) {
        let bucket: i32 = (nid >> self.bucket_shift())
            .try_into()
            .expect("Node id >> by bucket size is too large to fit in i32. This tool uses optimizations which assume that it will fit");

        let local_index = (nid % (2_i64.pow(self.bucket_shift() as u32))) as usize;
        (bucket, local_index)
    }

    /// Write the contents of the cache to the inner
    fn write_out_cache(&mut self) {
        let bucket_shift = self.bucket_shift();
        if let Some(cache) = &mut self.cache {
            let inner_entry = self
                .inner
                .entry(cache.0)
                .or_insert_with(|| Vec::with_capacity(8 + 2 * cache.1.len()));
            bucket_bytes_write(bucket_shift, &cache.1, inner_entry);
        }
    }

    /// Set the cache to be the value for this node id
    fn warm_cache(&mut self, nid: i64) {
        let (bucket_id, _local_index) = self.nodeid_bucket_local(nid);
        let bucket_shift = self.bucket_shift();

        // Do we have a cache that isn't for this node id? If so, write that out
        if let Some(cache) = &mut self.cache {
            if cache.0 != bucket_id {
                self.write_out_cache();
            }
        }

        // Have to duplicate it for lifetime reasons
        if let Some(cache) = &mut self.cache {
            if cache.0 != bucket_id {
                // now take from the real store to the cache.
                let bytes: &[u8] = self.inner.entry(bucket_id).or_insert_with(|| vec![0]);
                cache.0 = bucket_id;
                cache.1.truncate(0);
                cache.1.reserve(2_usize.pow(bucket_shift as u32));
                cache.1.extend(bucket_bytes_read(bucket_shift, bytes));
            }
        } else {
            // no cache, so store it
            // Read from the inner data
            let bytes: &[u8] = self.inner.entry(bucket_id).or_insert_with(|| vec![0]);
            let latlngs = bucket_bytes_read(self.bucket_shift, bytes).collect::<Vec<_>>();
            self.cache = Some((bucket_id, latlngs));
        }
    }
}

impl NodeIdPosition for NodeIdPositionBucket {
    fn new() -> Self {
        Self::with_bucket(6)
    }

    fn insert(&mut self, nid: i64, pos: (f64, f64)) {
        trace!("nodeid_pos.insert({}, ({}, {}))", nid, pos.0, pos.1);
        let pos: (i32, i32) = (
            Lat::try_from(pos.0).unwrap().inner(),
            Lon::try_from(pos.1).unwrap().inner(),
        );
        self.insert_i32(nid, pos);
    }

    fn insert_i32(&mut self, nid: i64, pos: (i32, i32)) {
        self.warm_cache(nid);
        let (_bucket_id, local_index) = self.nodeid_bucket_local(nid);

        let latlngs = &mut self.cache.as_mut().unwrap().1;

        if latlngs[local_index].is_none() {
            trace!("inc self.num_nodes");
            self.num_nodes += 1;
        } else {
            trace!("latlngs[{}] {:?}", local_index, latlngs[local_index]);
        }
        latlngs[local_index] = Some(pos);
    }

    fn get(&self, nid: &i64) -> Result<(f64, f64)> {
        let (bucket_id, local_index) = self.nodeid_bucket_local(*nid);
        if let Some((cache_bucket_id, cache_latlngs)) = &self.cache {
            if *cache_bucket_id == bucket_id {
                return cache_latlngs[local_index]
                    .map(|(lat_i32, lng_i32)| {
                        (
                            Lat::from_inner(lat_i32).degrees(),
                            Lon::from_inner(lng_i32).degrees(),
                        )
                    })
                    .ok_or_else(|| anyhow::anyhow!("Couldn't get node position for nid {nid}"));
            }
        }

        self.inner
            .get(&bucket_id)
            .and_then(|bytes| {
                bucket_bytes_read(self.bucket_shift, bytes)
                    .nth(local_index)
                    .unwrap()
            })
            .map(|(lat_i32, lng_i32)| {
                (
                    Lat::from_inner(lat_i32).degrees(),
                    Lon::from_inner(lng_i32).degrees(),
                )
            })
            .ok_or_else(|| anyhow::anyhow!("Couldn't get node position for nid {nid}"))
    }

    fn get_many_unwrap(&self, nids: &[i64], output: &mut [(f64, f64)]) {
        assert_eq!(nids.len(), output.len());
        if nids.is_empty() {
            return;
        }
        let mut buckets = HashMap::new();
        let mut bucket_id;
        let mut new_bucket_id;
        let mut bucket_latlngs_i32s;
        let mut local_index;
        let mut new_local_index;
        (bucket_id, local_index) = self.nodeid_bucket_local(nids[0]);

        bucket_latlngs_i32s = buckets.entry(bucket_id).or_insert_with(|| {
            let new_bucket_value = self.inner.get(&bucket_id).unwrap();
            bucket_bytes_read(self.bucket_shift, new_bucket_value).collect::<Vec<_>>()
        });
        assert!(
            bucket_latlngs_i32s[local_index].is_some(),
            "Node {} does not have a position",
            nids[0]
        );
        output[0] = bucket_latlngs_i32s[local_index]
            .map(|(lat_i32, lng_i32)| {
                (
                    Lat::from_inner(lat_i32).degrees(),
                    Lon::from_inner(lng_i32).degrees(),
                )
            })
            .unwrap();

        for (nid, output_el) in nids[1..].iter().zip(output[1..].iter_mut()) {
            (new_bucket_id, new_local_index) = self.nodeid_bucket_local(*nid);
            if new_bucket_id != bucket_id {
                bucket_latlngs_i32s = buckets.entry(new_bucket_id).or_insert_with(|| {
                    let new_bucket_value = self.inner.get(&new_bucket_id).unwrap();
                    bucket_bytes_read(self.bucket_shift, new_bucket_value).collect::<Vec<_>>()
                });
                bucket_id = new_bucket_id;
            }
            local_index = new_local_index;

            assert!(
                bucket_latlngs_i32s[local_index].is_some(),
                "Node {nid} does not have a position"
            );

            *output_el = bucket_latlngs_i32s[local_index]
                .map(|(lat_i32, lng_i32)| {
                    (
                        Lat::from_inner(lat_i32).degrees(),
                        Lon::from_inner(lng_i32).degrees(),
                    )
                })
                .unwrap_or_else(|| panic!("unable to get position for nid {nid}"));
        }
    }

    fn len(&self) -> usize {
        self.num_nodes
    }

    fn detailed_size(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Size of nodeid_pos (NodeIdPositionBucket): {} = {} bytes.\nnum_nodes: {} = {}.\nbytes/node = {:>.2}\ninner bucket_len {} = {}\nbucket_shift = {}",
            self.get_size(),
            self.get_size().to_formatted_string(&Locale::en),
            self.len(),
            self.len().to_formatted_string(&Locale::en),
            self.get_size() as f64 / self.len() as f64,
            self.inner.len(),
            self.inner.len().to_formatted_string(&Locale::en),
            self.bucket_shift(),
        ));
        output
    }

    fn finished_inserting(&mut self) {
        self.write_out_cache();
    }
}

// First i64 has the i-th bit set if there is a node at position i
// then there are all lat's delta encoded & varint encoded. then all the lng's
// TODO this could return an iterator
fn bucket_bytes_read(
    bucket_size: i64,
    bytes: &[u8],
) -> impl Iterator<Item = Option<(i32, i32)>> + '_ {
    assert!(bucket_size <= 6); // only support i64
    let (mask, bytes) = vartyint::read_i64(bytes).expect("");

    (0..2_i32.pow(bucket_size as u32)).scan(
        (0i32, 0i32, bytes),
        move |(curr_0, curr_1, bytes), i| {
            if (mask >> i) & 1 == 1 {
                let (n, new_bytes) = vartyint::read_i32(bytes).unwrap();
                *bytes = new_bytes;
                let p0 = *curr_0 + n;

                let (n, new_bytes) = vartyint::read_i32(bytes).unwrap();
                *bytes = new_bytes;
                let p1 = *curr_1 + n;
                *curr_0 = p0;
                *curr_1 = p1;
                Some(Some((p0, p1)))
            } else {
                Some(None)
            }
        },
    )
}

/// Store the node positions
/// Data format:
///   varint i64 bitmask. if bit i is set (i.e. `1`) then that node has a position set, `0` =
///   nid not in the bucket
///   Then 2N more varint32's. 2 int for each node, the latitude & longitudes
///   All lats are stored as the offset from the last lat. (lng are the same).
fn bucket_bytes_write(bucket_size: i64, pos: &[Option<(i32, i32)>], output: &mut Vec<u8>) {
    assert_eq!(pos.len(), 2_i32.pow(bucket_size as u32) as usize);
    assert!(bucket_size <= 6); // only support i64
    output.truncate(0);
    output.reserve(8 + 2 * pos.len());

    // Node id mask
    let mut mask = 0i64;
    for (i, p) in pos.iter().enumerate() {
        if p.is_some() {
            mask |= 1 << i;
        }
    }
    vartyint::write_i64(mask, output);

    // the locations
    let mut curr_0 = 0;
    let mut curr_1 = 0;
    for p in pos.iter().filter_map(|x| *x) {
        vartyint::write_i32(p.0 - curr_0, output);
        vartyint::write_i32(p.1 - curr_1, output);
        curr_0 = p.0;
        curr_1 = p.1;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! test_round_trip {
        ( $name:ident, $bucket_shift: expr_2021, $input:expr_2021 ) => {
            #[test]
            fn $name() {
                let input = $input;
                let bucket_shift: i64 = $bucket_shift;
                let mut output = vec![];
                bucket_bytes_write(bucket_shift, &input, &mut output);
                let result = bucket_bytes_read(bucket_shift, &output).collect::<Vec<_>>();

                assert_eq!(
                    result, input,
                    "Ouput was {:?} but expected {:?}",
                    result, input,
                );
            }
        };
    }

    test_round_trip!(empty, 2, vec![None; 4]);
    test_round_trip!(test1, 2, vec![Some((1, 1)), None, None, None]);
    test_round_trip!(test2, 2, vec![Some((1, 1)), Some((1, 1)), None, None]);
    test_round_trip!(test3, 2, vec![Some((2, 2)), None, None, None]);
    test_round_trip!(test4, 2, vec![Some((2, 2)), Some((100, 100)), None, None]);
    test_round_trip!(
        test5,
        2,
        vec![
            Some((2, 2)),
            Some((100, 100)),
            Some((100, 100)),
            Some((2, 2))
        ]
    );
    test_round_trip!(test6, 2, vec![None, None, None, Some((1, 1))]);

    //fn init() {
    //    let _ = env_logger::builder().is_test(true).try_init();
    //}
    //#[test]
    //fn real_life() {
    //    init();
    //    let mut nodeid_pos = NodeIdPositionBucket::with_bucket(5);
    //    assert_eq!(nodeid_pos.len(), 0);

    //    nodeid_pos.insert(1494342551, (14.2637113, 36.0239228));
    //    assert_eq!(nodeid_pos.get(&1494342551), Some((14.2637113, 36.0239228)));
    //    assert_eq!(nodeid_pos.len(), 1);
    //    nodeid_pos.insert(1494342553, (14.2663291, 36.0216147));
    //    assert_eq!(nodeid_pos.get(&1494342553), Some((14.2663291, 36.0216147)));
    //    assert_eq!(nodeid_pos.len(), 2);
    //    nodeid_pos.insert(1494342577, (14.2647091, 36.0225345));
    //    assert_eq!(nodeid_pos.get(&1494342577), Some((14.2647091, 36.0225345)));
    //    assert_eq!(nodeid_pos.len(), 3);
    //    nodeid_pos.insert(1494342562, (14.2627028, 36.0234716));
    //    assert_eq!(nodeid_pos.get(&1494342562), Some((14.2627028, 36.0234716)));
    //    assert_eq!(nodeid_pos.len(), 4);
    //    nodeid_pos.insert(1494342554, (14.265385, 36.0215626));
    //    assert_eq!(nodeid_pos.get(&1494342554), Some((14.265385, 36.0215626)));
    //    assert_eq!(nodeid_pos.len(), 5);
    //    nodeid_pos.insert(1494342589, (14.2580679, 36.0263698));
    //    assert_eq!(nodeid_pos.get(&1494342589), Some((14.2580679, 36.0263698)));
    //    assert_eq!(nodeid_pos.len(), 6);
    //    nodeid_pos.insert(1494342590, (14.2643657, 36.0229076));
    //    assert_eq!(nodeid_pos.get(&1494342590), Some((14.2643657, 36.0229076)));
    //    assert_eq!(nodeid_pos.len(), 7);
    //    nodeid_pos.insert(1494342591, (14.2647842, 36.0228035));
    //    assert_eq!(nodeid_pos.get(&1494342591), Some((14.2647842, 36.0228035)));
    //    assert_eq!(nodeid_pos.len(), 8);
    //    nodeid_pos.insert(1494342598, (14.2646447, 36.0222047));
    //    assert_eq!(nodeid_pos.get(&1494342598), Some((14.2646447, 36.0222047)));
    //    assert_eq!(nodeid_pos.len(), 9);
    //    nodeid_pos.insert(1494342550, (14.2670694, 36.0217622));
    //    assert_eq!(nodeid_pos.get(&1494342550), Some((14.2670694, 36.0217622)));
    //    assert_eq!(nodeid_pos.len(), 10);
    //    nodeid_pos.insert(1494342579, (14.2648593, 36.0226733));
    //    assert_eq!(nodeid_pos.get(&1494342579), Some((14.2648593, 36.0226733)));
    //    assert_eq!(nodeid_pos.len(), 11);
    //    nodeid_pos.insert(1494342567, (14.2602673, 36.0260487));
    //    assert_eq!(nodeid_pos.get(&1494342567), Some((14.2602673, 36.0260487)));
    //    assert_eq!(nodeid_pos.len(), 12);
    //    nodeid_pos.insert(1494342569, (14.2677131, 36.0217709));
    //    assert_eq!(nodeid_pos.get(&1494342569), Some((14.2677131, 36.0217709)));
    //    assert_eq!(nodeid_pos.len(), 13);
    //    nodeid_pos.insert(1494342582, (14.263958, 36.0238534));
    //    assert_eq!(nodeid_pos.get(&1494342582), Some((14.263958, 36.0238534)));
    //    assert_eq!(nodeid_pos.len(), 14);
    //    nodeid_pos.insert(1494342568, (14.2647198, 36.0219357));
    //    assert_eq!(nodeid_pos.get(&1494342568), Some((14.2647198, 36.0219357)));
    //    assert_eq!(nodeid_pos.len(), 15);
    //    nodeid_pos.insert(1494342557, (14.2693198, 36.020732));
    //    assert_eq!(nodeid_pos.get(&1494342557), Some((14.2693198, 36.020732)));
    //    assert_eq!(nodeid_pos.len(), 16);
    //    nodeid_pos.insert(1494342599, (14.2626062, 36.0252331));
    //    assert_eq!(nodeid_pos.get(&1494342599), Some((14.2626062, 36.0252331)));
    //    assert_eq!(nodeid_pos.len(), 17);
    //    nodeid_pos.insert(1494342602, (14.2533043, 36.0282613));
    //    assert_eq!(nodeid_pos.get(&1494342602), Some((14.2533043, 36.0282613)));
    //    assert_eq!(nodeid_pos.len(), 18);
    //    nodeid_pos.insert(1494342600, (14.2689469, 36.0209031));
    //    assert_eq!(nodeid_pos.get(&1494342600), Some((14.2689469, 36.0209031)));
    //    assert_eq!(nodeid_pos.len(), 19);
    //    nodeid_pos.insert(1494342611, (14.258894, 36.0261268));
    //    assert_eq!(nodeid_pos.get(&1494342611), Some((14.258894, 36.0261268)));
    //    assert_eq!(nodeid_pos.len(), 20);
    //    nodeid_pos.insert(1494342618, (14.2593876, 36.0258405));
    //    assert_eq!(nodeid_pos.get(&1494342618), Some((14.2593876, 36.0258405)));
    //    assert_eq!(nodeid_pos.len(), 21);
    //    nodeid_pos.insert(1494342625, (14.2557076, 36.0266387));
    //    assert_eq!(nodeid_pos.get(&1494342625), Some((14.2557076, 36.0266387)));
    //    assert_eq!(nodeid_pos.len(), 22);
    //    nodeid_pos.insert(1494342646, (14.2640331, 36.0231679));
    //    assert_eq!(nodeid_pos.get(&1494342646), Some((14.2640331, 36.0231679)));
    //    assert_eq!(nodeid_pos.len(), 23);
    //    nodeid_pos.insert(1494342647, (14.2682496, 36.0215886));
    //    assert_eq!(nodeid_pos.get(&1494342647), Some((14.2682496, 36.0215886)));
    //    assert_eq!(nodeid_pos.len(), 24);
    //    nodeid_pos.insert(1494342648, (14.2686251, 36.0211982));
    //    assert_eq!(nodeid_pos.get(&1494342648), Some((14.2686251, 36.0211982)));
    //    assert_eq!(nodeid_pos.len(), 25);
    //    nodeid_pos.insert(1494342650, (14.2679921, 36.0217448));
    //    assert_eq!(nodeid_pos.get(&1494342650), Some((14.2679921, 36.0217448)));
    //    assert_eq!(nodeid_pos.len(), 26);
    //    nodeid_pos.insert(1494342658, (14.2571345, 36.0264131));
    //    assert_eq!(nodeid_pos.get(&1494342658), Some((14.2571345, 36.0264131)));
    //    assert_eq!(nodeid_pos.len(), 27);
    //    nodeid_pos.insert(1494342612, (14.2627457, 36.024027));
    //    assert_eq!(nodeid_pos.get(&1494342612), Some((14.2627457, 36.024027)));
    //    assert_eq!(nodeid_pos.len(), 28);
    //    nodeid_pos.insert(1494342664, (14.2622629, 36.0255194));
    //    assert_eq!(nodeid_pos.get(&1494342664), Some((14.2622629, 36.0255194)));
    //    assert_eq!(nodeid_pos.len(), 29);
    //    nodeid_pos.insert(1494342665, (14.2549458, 36.0268904));
    //    assert_eq!(nodeid_pos.get(&1494342665), Some((14.2549458, 36.0268904)));
    //    assert_eq!(nodeid_pos.len(), 30);
    //    nodeid_pos.insert(1494342644, (14.2629603, 36.0231506));
    //    assert_eq!(nodeid_pos.get(&1494342644), Some((14.2629603, 36.0231506)));
    //    assert_eq!(nodeid_pos.len(), 31);
    //    nodeid_pos.insert(1494342607, (14.2608574, 36.0261008));
    //    assert_eq!(nodeid_pos.get(&1494342607), Some((14.2608574, 36.0261008)));
    //    assert_eq!(nodeid_pos.len(), 32);
    //    nodeid_pos.insert(1494342615, (14.2544416, 36.0271073));
    //    assert_eq!(nodeid_pos.get(&1494342615), Some((14.2544416, 36.0271073)));
    //    assert_eq!(nodeid_pos.len(), 33);
    //    nodeid_pos.insert(1494342620, (14.2640117, 36.0234109));
    //    assert_eq!(nodeid_pos.get(&1494342620), Some((14.2640117, 36.0234109)));
    //    assert_eq!(nodeid_pos.len(), 34);
    //    nodeid_pos.insert(1494342662, (14.2633679, 36.0231506));
    //    assert_eq!(nodeid_pos.get(&1494342662), Some((14.2633679, 36.0231506)));
    //    assert_eq!(nodeid_pos.len(), 35);
    //    nodeid_pos.insert(1494342604, (14.263604, 36.0234976));
    //    assert_eq!(nodeid_pos.get(&1494342604), Some((14.263604, 36.0234976)));
    //    assert_eq!(nodeid_pos.len(), 36);
    //    nodeid_pos.insert(1494342614, (14.253948, 36.0274804));
    //    assert_eq!(nodeid_pos.get(&1494342614), Some((14.253948, 36.0274804)));
    //    assert_eq!(nodeid_pos.len(), 37);
    //    nodeid_pos.insert(1494342675, (14.2628744, 36.0248773));
    //    assert_eq!(nodeid_pos.get(&1494342675), Some((14.2628744, 36.0248773)));
    //    assert_eq!(nodeid_pos.len(), 38);
    //    nodeid_pos.insert(3933446907, (14.2571911, 36.026411));
    //    assert_eq!(nodeid_pos.get(&3933446907), Some((14.2571911, 36.026411)));
    //    assert_eq!(nodeid_pos.len(), 39);
    //    nodeid_pos.insert(4008848336, (14.2623883, 36.0254219));
    //    assert_eq!(nodeid_pos.get(&4008848336), Some((14.2623883, 36.0254219)));
    //    assert_eq!(nodeid_pos.len(), 40);
    //}
}
