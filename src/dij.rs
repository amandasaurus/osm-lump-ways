#![allow(clippy::type_complexity, dead_code)]
use super::*;
use anyhow::Result;
use graph::UndirectedAdjGraph;
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;
use way_group::WayGroup;

#[derive(Clone, Debug, Default, clap::ValueEnum)]
pub(crate) enum SplitPathsMethod {
    #[default]
    LongestPath,
    AsCrowFlies,
}

fn min_max<T: PartialOrd>(a: T, b: T) -> (T, T) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

pub(crate) fn into_segments(
    wg: &WayGroup,
    nodeid_pos: &(impl NodeIdPosition + std::marker::Send),
    min_length_m: Option<f64>,
    only_longest_n_splitted_paths: Option<usize>,
    max_sinuosity: Option<f64>,
    split_paths_method: SplitPathsMethod,
    splitter: &ProgressBar,
) -> Result<Vec<Vec<i64>>> {
    let nodeid_pos = Arc::new(Mutex::new(nodeid_pos));
    let mut results = Vec::new();

    trace!(
        "Starting into_segments with wg.num_nodeids {}",
        wg.num_nodeids()
    );
    let mut edges = UndirectedAdjGraph::new();
    let (sender, receiver) = std::sync::mpsc::channel();
    wg.nodeids
        .par_iter()
        .flat_map(|coord_string| coord_string.par_windows(2))
        // There are a few cases where a node is doubled in a way, and this code requires that they
        // be ordered.
        .filter(|edge| edge[0] != edge[1])
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

            (min_max(raw_edge[0], raw_edge[1]), dist as f32)
        })
        .for_each_with(sender, |sender, x| sender.send(x).unwrap());

    for ((nid1, nid2), dist) in receiver.iter() {
        assert!(
            nid1 < nid2,
            "Creating the Dij graph, and got nid1<nid2 nid1={nid1} nid2={nid2} dist={dist}"
        );
        edges.set(&nid1, &nid2, dist);
    }

    let old = (edges.num_edges(), edges.num_vertexes());
    if max_sinuosity.is_none() {
        edges.contract_edges();
        debug!(
            "wg:{} Post-contraction. Removed {} edges and {} vertexes",
            wg.root_wayid,
            old.0 - edges.num_edges(),
            old.1 - edges.num_vertexes()
        );
        // This vertexes are now “done”
        splitter.inc((old.1 - edges.num_vertexes()) as u64);
    }

    for paths_generated_so_far in 0..only_longest_n_splitted_paths.unwrap_or(1_000_000) {
        if edges.is_empty() {
            // graph empty. Nothing to do
            continue;
        }
        trace!(
            "wg:{} step:{} Starting Dij alg step. We have {} vertexes & {} edges",
            wg.root_wayid,
            paths_generated_so_far,
            edges.num_vertexes(),
            edges.num_edges(),
        );

        if edges.len() < 50 {
            trace!(
                "wg:{} Input edges:\n{}",
                wg.root_wayid,
                edges.pretty_print(
                    &|el| if el.is_nan() {
                        "_".to_string()
                    } else {
                        "#".to_string()
                    },
                    ""
                )
            );
        }

        // Current longest route. (startnid, endnid, prevnid, distance)
        let mut longest_summary: Option<(i64, i64, Option<i64>, f32)> = None;
        // The actual graph of the longest
        let mut longest_graph: Option<HashMap<i64, (Option<i64>, OrderedFloat<f32>)>> = None;
        log!(
            if edges.len() > 10_000 { Debug } else { Trace },
            "wg:{} about to start dij_single. edges.len() {}",
            wg.root_wayid,
            edges.len()
        );
        if edges.len() > 10_000 {
            trace!(
                "wg:{} doing this single threaded because it's so big ({} edges)",
                wg.root_wayid,
                edges.len()
            );
            // do it single threaded, Can't handle the memory requirements
            let mut this_is_longest;
            let (mut p1, mut p2);
            for (nid1, results_from_nid1) in edges
                .vertexes()
                .map(|nid1| (nid1, dij_single(*nid1, &edges)))
            {
                this_is_longest = false; // should we save the graph of distances?
                p1 = nodeid_pos.lock().unwrap().get(nid1).unwrap();
                for (nid2, (prev_nid, dist)) in results_from_nid1.iter().filter(|x| x.0 != nid1) {
                    p2 = nodeid_pos.lock().unwrap().get(nid2).unwrap();
                    let dist_path = dist.into_inner();
                    let dist_ends = haversine_m(p1.1, p1.0, p2.1, p2.0) as f32;
                    let relevant_distance = match split_paths_method {
                        SplitPathsMethod::LongestPath => dist_path,
                        SplitPathsMethod::AsCrowFlies => dist_ends,
                    };

                    // TODO continue if nid1 > nid2?
                    if longest_summary.map_or(false, |(_, _, _, dist_longest_so_far)| {
                        relevant_distance < dist_longest_so_far
                    }) {
                        // this path is not longer than the previous longest we've accepted. So
                        // skip it
                        continue;
                    }

                    if let Some(max_sinuosity) = max_sinuosity {
                        // we only include it if the sinuosity is not too long
                        // get local id for the node ids
                        p2 = nodeid_pos.lock().unwrap().get(nid2).unwrap();
                        let dist_path = dist.into_inner() as f64;
                        let dist_ends = haversine_m(p1.1, p1.0, p2.1, p2.0);
                        if (dist_path / dist_ends) <= max_sinuosity {
                            longest_summary = Some((*nid1, *nid2, *prev_nid, relevant_distance));
                            this_is_longest = true;
                        } else {
                            // This is path has too high a sinuosity, so skip it
                        }
                    } else {
                        // we don't care about sinuosity, so save it
                        longest_summary = Some((*nid1, *nid2, *prev_nid, relevant_distance));
                        this_is_longest = true;
                    }
                }
                if this_is_longest || longest_graph.is_none() {
                    longest_graph = Some(results_from_nid1);
                }
            }
        } else {
            trace!(
                "wg:{} doing this multithreaded because it's not so big ({} edges)",
                wg.root_wayid,
                edges.len()
            );
            let (sender, receiver) = std::sync::mpsc::channel();
            // multithreaded
            edges
                .vertexes()
                .par_bridge()
                .map(|nid1| (nid1, dij_single(*nid1, &edges)))
                .for_each_with(sender, |s, x| s.send(x).unwrap());

            // look for the longest distance

            let mut this_is_longest: bool;
            let (mut p1, mut p2);
            for (nid1, results_from_nid1) in receiver.iter() {
                p1 = nodeid_pos.lock().unwrap().get(nid1).unwrap();
                this_is_longest = false; // should we save this graph, because it's the longest

                // TODO replace this with one big iterator and use `.max_by()` to get the “longest”
                for (nid2, (prev_nid, dist)) in results_from_nid1.iter().filter(|x| x.0 != nid1) {
                    p2 = nodeid_pos.lock().unwrap().get(nid2).unwrap();
                    let dist_path = dist.into_inner();
                    let dist_ends = haversine_m(p1.1, p1.0, p2.1, p2.0) as f32;
                    let relevant_distance = match split_paths_method {
                        SplitPathsMethod::LongestPath => dist_path,
                        SplitPathsMethod::AsCrowFlies => dist_ends,
                    };

                    // TODO continue if nid1 > nid2?
                    if longest_summary.map_or(false, |(_, _, _, dist_longest_so_far)| {
                        relevant_distance < dist_longest_so_far
                    }) {
                        // this path is not longer than the previous longest we've accepted. So
                        // skip it
                        continue;
                    }

                    if let Some(max_sinuosity) = max_sinuosity {
                        // we only include it if the sinuosity is not too long
                        // get local id for the node ids
                        if (dist_path / dist_ends) <= max_sinuosity as f32 {
                            longest_summary = Some((*nid1, *nid2, *prev_nid, relevant_distance));
                            this_is_longest = true;
                        } else {
                            // This is path has too high a sinuosity, so skip it
                        }
                    } else {
                        // we don't care about sinuosity, so save it
                        longest_summary = Some((*nid1, *nid2, *prev_nid, relevant_distance));
                        this_is_longest = true;
                    }
                }
                if this_is_longest || longest_graph.is_none() {
                    longest_graph = Some(results_from_nid1);
                }
            }
        }
        let longest_summary = longest_summary.expect("WTF no longest");
        let longest_graph = longest_graph.expect("The longest graph should have been set");

        trace!(
            "wg:{} Longest is of length {:.1} and from {} → {} (via {:?})",
            wg.root_wayid,
            longest_summary.3,
            longest_summary.0,
            longest_summary.1,
            longest_summary.2
        );

        if let Some(min_length_m) = min_length_m {
            if longest_summary.3 < min_length_m as f32 {
                trace!(
                    "wg:{} longest is < {}, so skipping the rest",
                    wg.root_wayid,
                    min_length_m
                );
                splitter.inc(edges.num_vertexes() as u64); // these vertexes are “done”
                break;
            }
        }

        // Build the path of nodeids (using the contracted edges)
        // we know the end nid, and the second last one. So build the path in reverse
        let first_pos = longest_summary.0;
        let mut last_pos = longest_summary.1;
        let mut path = vec![last_pos];
        while first_pos != last_pos {
            last_pos = longest_graph.get(&last_pos).expect("").0.unwrap();
            path.push(last_pos);
        }
        splitter.inc(path.len() as u64);
        trace!(
            "wg:{} The contracted path has {} nodes (reversed:) {:?}",
            wg.root_wayid,
            path.len(),
            path
        );

        // Turn the local nids into proper node ids. do it in reverse
        path.reverse();

        let mut full_path = vec![];
        for a_b in path.windows(2) {
            full_path.push(a_b[0]);
            full_path.extend(edges.get_intermediates(&a_b[0], &a_b[1]).unwrap());
        }
        // and the last one
        full_path.push(*path.last().unwrap());

        let old_num_edges = edges.len();
        for a_b in path.windows(2) {
            edges.remove_edge(&a_b[0], &a_b[1]);
        }
        trace!(
            "wg:{} Post-removing used edges: There are now {} edges in this graph, reduced by {} from {}",
            wg.root_wayid,
            edges.len(),
            old_num_edges - edges.len(),
            old_num_edges
        );
        let old = (edges.num_edges(), edges.num_vertexes());
        if max_sinuosity.is_none() {
            edges.contract_edges();
            debug!(
                "wg:{} Post-contraction. Removed {} edges and {} vertexes",
                wg.root_wayid,
                old.0 - edges.num_edges(),
                old.1 - edges.num_vertexes()
            );
        }

        results.push(full_path);
    }
    log!(
        if results.len() > 5 { Debug } else { Trace },
        "wg:{} Finished into_segments. From {} nodes, found {} paths",
        wg.root_wayid,
        wg.num_nodeids().to_formatted_string(&Locale::en),
        results.len()
    );

    Ok(results)
}

/// Does a single Dijkstra search from start_idx to all vertexes.
/// return a hashmap for each other vertex in the graph, with the previous node to go to (if
/// applicable) and the total distance along the path
fn dij_single(
    start_idx: i64,
    edges: &UndirectedAdjGraph<i64, f32>,
) -> HashMap<i64, (Option<i64>, OrderedFloat<f32>)> {
    trace!(
        "dij_single started. start_idx {} edges.len() {}",
        start_idx,
        edges.len()
    );
    let mut prev_dist = HashMap::new();
    prev_dist
        .try_reserve(edges.num_vertexes())
        .expect("Unable to reserve enough space for Dij. alg.");
    prev_dist.insert(start_idx, (None, OrderedFloat(0.)));

    let mut frontier = BinaryHeap::new();
    // hack to just store negative distance so the shortest distance is the largest number
    frontier.push((OrderedFloat(-0.), start_idx));

    let mut this_dist;
    while let Some((mut curr_dist, curr_id)) = frontier.pop() {
        curr_dist *= -1.;
        //trace!(
        //    "Current frontier. id {:>13} curr dist {:>8}, currently shortest known: {:>10} frontier.len() {:>3}",
        //    curr_id,
        //    curr_dist,
        //    prev_dist[&curr_id].1,
        //    frontier.len()
        //);
        if curr_dist > prev_dist[&curr_id].1 {
            // already found a shorter
            continue;
        }
        for neighbor in edges
            .neighbors(&curr_id)
            .map(|(i, d)| (i, OrderedFloat(*d)))
        {
            this_dist = neighbor.1 + curr_dist;
            prev_dist
                .entry(*neighbor.0)
                .and_modify(|(prev_id, dist)| {
                    if this_dist < *dist {
                        *prev_id = Some(curr_id);
                        *dist = this_dist;
                        frontier.push((-this_dist, *neighbor.0));
                    }
                })
                .or_insert_with(|| {
                    frontier.push((-this_dist, *neighbor.0));
                    (Some(curr_id), this_dist)
                });
            //trace!("Now prev_dist.len {}", prev_dist.len());
        }
    }

    let results = prev_dist;

    trace!(
        "dij_single finished. start_idx {} edges.len() {} results.len() {}",
        start_idx,
        edges.len(),
        results.len()
    );

    results
}

///// Return the first cycle we find, starting at this vertex
//pub fn look_for_loop(start_vertex: i64, g: &DirectedGraph2) -> Option<Vec<i64>> {
//    trace!(
//        "dij_single started. start_vertex {} g.len() {:?}",
//        start_vertex,
//        g.len()
//    );
//    let mut prev_dist = HashMap::new();
//    prev_dist.insert(start_vertex, (None, 0));
//
//    let mut frontier = BinaryHeap::new();
//    // hack to just store negative distance so the shortest distance is the largest number
//    frontier.push((-0_i64, start_vertex));
//
//    let mut this_dist;
//    while let Some((mut curr_dist, curr_id)) = frontier.pop() {
//        curr_dist *= -1;
//        //trace!(
//        //    "Current frontier. id {:>13} curr dist {:>8}, currently shortest known: {:>10} frontier.len() {:>3}",
//        //    curr_id,
//        //    curr_dist,
//        //    prev_dist[&curr_id].1,
//        //    frontier.len()
//        //);
//        if curr_dist > prev_dist[&curr_id].1 {
//            // already found a shorter
//            continue;
//        }
//
//        for neighbor in g.neighbors(&curr_id) {
//            if *neighbor == start_vertex {
//                // found a cycle!
//                // build the cycle (but backwards)
//                let mut cycle_path = Vec::with_capacity(curr_dist as usize + 1);
//                cycle_path.push(*neighbor);
//                let mut path_id = curr_id;
//                cycle_path.push(path_id);
//                loop {
//                    cycle_path.push(path_id);
//                    if path_id == start_vertex {
//                        break;
//                    }
//                    path_id = prev_dist[&path_id].0.unwrap();
//                }
//                cycle_path.reverse();
//
//                return Some(cycle_path);
//            }
//            this_dist = curr_dist + 1;
//            prev_dist
//                .entry(*neighbor)
//                .and_modify(|(prev_id, dist)| {
//                    if this_dist < *dist {
//                        *prev_id = Some(curr_id);
//                        *dist = this_dist;
//                        frontier.push((-this_dist, *neighbor));
//                    }
//                })
//                .or_insert_with(|| {
//                    frontier.push((-this_dist, *neighbor));
//                    (Some(curr_id), this_dist)
//                });
//            //trace!("Now prev_dist.len {}", prev_dist.len());
//        }
//    }
//
//    // gotten to here without anything
//    None
//}
