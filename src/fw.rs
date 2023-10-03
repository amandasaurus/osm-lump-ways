/// https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
use super::*;
use anyhow::{Context, Result};
use graph::{DirectedGraph, UndirectedGraph};
use std::iter;

fn min_max<T: PartialOrd>(a: T, b: T) -> (T, T) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

pub(crate) fn into_fw_segments(
    wg: &WayGroup,
    nodeid_pos: &(impl NodeIdPosition + std::marker::Send),
    min_length_m: Option<f64>,
    only_longest_n_splitted_paths: Option<usize>,
) -> Result<Vec<Vec<i64>>> {
    let mut results = Vec::new();

    let mut orig_edges = HashMap::with_capacity(wg.num_nodeids());

    let nodeid_pos = Arc::new(Mutex::new(nodeid_pos));

    orig_edges.par_extend(
        wg.nodeids
            .par_iter()
            .flat_map(|coord_string| coord_string.par_windows(2))
            .map_with(nodeid_pos.clone(), |nodeid_pos, edge| {
                if nodeid_pos.lock().unwrap().contains_key(&edge[0])
                    && nodeid_pos.lock().unwrap().contains_key(&edge[1])
                {
                    Some(edge)
                } else {
                    warn!("No position found for edge: {:?}", edge);
                    None
                }
            })
            .filter_map(|x| x)
            .map_with(nodeid_pos.clone(), |nodeid_pos, raw_edge| {
                // get local id for the node ids
                let p1 = nodeid_pos.lock().unwrap().get(&raw_edge[0]).unwrap();
                let p2 = nodeid_pos.lock().unwrap().get(&raw_edge[1]).unwrap();
                let dist = haversine_m(p1.1, p1.0, p2.1, p2.0);

                (min_max(raw_edge[0], raw_edge[1]), dist)
            }),
    );

    let mut nodeids: Vec<i64> = Vec::with_capacity(orig_edges.len());

    for paths_generated_so_far in 0..only_longest_n_splitted_paths.unwrap_or(1_000_000) {
        if orig_edges.is_empty() {
            trace!(
                "wg:{} step:{} Graph is empty, so finished",
                wg.root_wayid,
                paths_generated_so_far
            );
            break;
        }
        nodeids.truncate(0);
        nodeids.extend(
            orig_edges
                .keys()
                .flat_map(|k| iter::once(k.0).chain(iter::once(k.1))),
        );
        nodeids.par_sort();
        nodeids.dedup();
        nodeids.shrink_to_fit();
        trace!(
            "wg:{} step:{} Starting FW alg step. We have {} vertexes & {} edges",
            wg.root_wayid,
            paths_generated_so_far,
            nodeids.len(),
            orig_edges.len()
        );
        // N×N array with Option(Distance)
        // Rather than using Option<f32>, we're (ab)using f32:NAN instead of None
        // nan → there is no edge here.
        let mut edges = UndirectedGraph::new(nodeids.len(), f32::NAN)
            .context("Creating initial edges graph")?;
        let mut i;
        let mut j;
        for ((p1, p2), dist) in orig_edges.iter() {
            i = nodeids.binary_search(p1).unwrap();
            j = nodeids.binary_search(p2).unwrap();
            edges.set(i, j, *dist as f32);
        }

        trace!(
            "wg:{} step:{} Starting FW alg",
            wg.root_wayid,
            paths_generated_so_far
        );

        trace!(
            "wg:{} Input edges:\n{}",
            wg.root_wayid,
            edges.pretty_print(
                &|el| if el.is_nan() {
                    ".".to_string()
                } else {
                    "#".to_string()
                },
                " "
            )
        );

        // Run FW on the new graph
        trace!(
            "wg:{} step:{} Starting FW alg",
            wg.root_wayid,
            paths_generated_so_far
        );
        let (fw_result_dist, fw_result_next) = fw(&edges);
        trace!(
            "wg:{} step:{} Finished FW alg",
            wg.root_wayid,
            paths_generated_so_far
        );
        // Find the longest

        // build the longest path
        let longest = fw_result_dist
            .values()
            .filter(|x| !x.2.is_infinite())
            .max_by(|a, b| a.2.total_cmp(b.2))
            .expect("No max");
        if let Some(min_length_m) = min_length_m {
            if (*longest.2 as f64) < min_length_m {
                // Longest is too short. Don't bother any more
                trace!(
                    "wg:{} step:{} Longest path is {:.1} which is less than the min of {}. Finished with this WG.",
                    wg.root_wayid, paths_generated_so_far,
                    longest.2, min_length_m
                );
                break;
            }
        }
        trace!(
            "wg:{} Longest is of length {:.1} and from {} → {}",
            wg.root_wayid,
            longest.2,
            longest.0,
            longest.1
        );
        trace!(
            "wg:{} Next's:\n{}",
            wg.root_wayid,
            fw_result_next.pretty_print(
                &|el| el.map_or_else(|| "-".to_string(), |e| e.to_string()),
                " "
            )
        );

        // Build the path of nodeids
        let mut curr_pos = longest.0;
        let last_point = longest.1;
        trace!("want to go from {} to {}", curr_pos, last_point);
        let mut path = vec![curr_pos];
        while curr_pos != last_point {
            trace!("Current path: {:?}", path);
            trace!(
                "Path from {} to {}: {:?}",
                curr_pos,
                last_point,
                fw_result_next.get(curr_pos, last_point)
            );
            curr_pos = fw_result_next.get(curr_pos, last_point).unwrap();
            trace!("Next location: {}", curr_pos);
            if path.contains(&curr_pos) {
                // This shouldn't happen?!
                warn!("Path {:?} curr_pos {}", path, curr_pos);
                panic!("wtf");
            }
            path.push(curr_pos);
        }
        trace!(
            "wg:{} The path has {} nodes {:?}",
            wg.root_wayid,
            path.len(),
            path
        );

        let path = path.into_iter().map(|i| nodeids[i]).collect::<Vec<_>>();

        let old_num_edges = orig_edges.len();
        for a_b in path.windows(2) {
            orig_edges.remove(&min_max(a_b[0], a_b[1]));
        }
        orig_edges.shrink_to_fit();
        trace!(
            "wg:{} There are now {} edges in this graph, reduced by {} from {}",
            wg.root_wayid,
            orig_edges.len(),
            old_num_edges - orig_edges.len(),
            old_num_edges
        );

        results.push(path);
    }

    trace!(
        "wg:{} Finished into_fw_segments, found {} paths",
        wg.root_wayid,
        results.len()
    );
    Ok(results)
}

fn fw(edges: &UndirectedGraph<f32>) -> (DirectedGraph<f32>, DirectedGraph<Option<usize>>) {
    trace!("FW start with {} edges", edges.len());
    let size = edges.len();

    let mut nexts = DirectedGraph::new(size, None);
    let mut distance = DirectedGraph::new(size, f32::INFINITY);

    for (i, j, len) in edges.values().filter(|e| !e.2.is_nan()) {
        trace!("Edge from {} to {} of len {:?}", i, j, len);
        distance.set(i, j, *len);
        nexts.set(i, j, Some(j));
        //nexts.set(j, i, Some(i));
    }

    for i in 0..size {
        distance.set(i, i, 0.);
        nexts.set(i, i, Some(i));
    }
    trace!(
        "Next (inside FW):\n{}",
        nexts.pretty_print(
            &|el| el.map_or_else(|| "-".to_string(), |e| e.to_string()),
            " "
        )
    );
    for k in 0..size {
        for i in 0..size {
            for j in 0..size {
                if *distance.get(i, j) > distance.get(i, k) + distance.get(k, j) {
                    distance.set(i, j, distance.get(i, k) + distance.get(k, j));
                    nexts.set(i, j, nexts.get(i, k).to_owned());
                }
            }
        }
    }

    (distance, nexts)
}
