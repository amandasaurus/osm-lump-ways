#![allow(clippy::type_complexity, dead_code)]
use super::*;
use graph::{DirectedGraph2, DirectedGraphTrait, Graph2, UndirectedAdjGraph};
use haversine::haversine_m_fpair;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;

#[derive(Clone, Debug, Default, clap::ValueEnum, PartialEq)]
pub enum SplitPathsMethod {
    #[default]
    AsCrowFlies,
    LongestPath,
}

fn min_max<T: PartialOrd>(a: T, b: T) -> (T, T) {
    if a < b { (a, b) } else { (b, a) }
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

/// Does a single A* search from start_idx to a set of other vertexes
pub(crate) fn paths_one_to_many<'a>(
    start: (i64, (OrderedFloat<f64>, OrderedFloat<f64>)),
    targets: &'a [(i64, (OrderedFloat<f64>, OrderedFloat<f64>))],
    nodeid_pos: &'a impl NodeIdPosition,
    edges: &'a Graph2,
) -> impl ParallelIterator<Item = ((i64, i64), Vec<i64>)> + 'a {
    let (start_idx, _start_pos) = start;
    assert!(edges.contains_vertex(start_idx));
    assert!(
        targets
            .par_iter()
            .map(|(i, _)| i)
            .all(|i| edges.contains_vertex(*i))
    );

    // TODO keep best_dist_prev between each targetted run

    //let mut results = Vec::with_capacity(targets.len());

    targets
        .par_iter()
        .map(move |target| path_one_to_one(start, *target, nodeid_pos, edges))
}

/// Does a single A* search from start_idx to another
pub(crate) fn path_one_to_one<'a>(
    start: (i64, (OrderedFloat<f64>, OrderedFloat<f64>)),
    target: (i64, (OrderedFloat<f64>, OrderedFloat<f64>)),
    nodeid_pos: &'a impl NodeIdPosition,
    edges: &'a Graph2,
) -> ((i64, i64), Vec<i64>) {
    let (start_idx, start_pos) = start;
    let (target_idx, target_pos) = target;
    let mut frontier = BinaryHeap::new();
    let mut best_dist_prev: HashMap<i64, (OrderedFloat<f64>, i64)> = HashMap::new();
    best_dist_prev.clear();
    best_dist_prev.insert(start_idx, (OrderedFloat(0.), start_idx));
    frontier.clear();
    frontier.push((
        OrderedFloat(-haversine_m(*start_pos.0, *start_pos.1, *target_pos.0, *target_pos.1) + 0.),
        OrderedFloat(-0.),
        start_idx,
        start_pos,
    ));

    let mut this_luft_dist = Default::default();
    let mut neighbor_pos_raw = Default::default();
    let mut neighbor_pos = Default::default();
    let mut this_path_dist;

    while let Some((_est_dist, mut curr_path_dist, curr_id, curr_pos)) = frontier.pop() {
        //dbg!(frontier.len()+1);
        curr_path_dist *= -1.;
        //info!("Frontier size {}, curr_id {}, est_dist {}, curr_path_dist {}", frontier.len()+1, curr_id, _est_dist, curr_path_dist);
        if curr_id == target_idx {
            assert!(
                best_dist_prev.contains_key(&target_idx),
                "Something went wrong with nid {}",
                target_idx
            );
            // finished
            break;
        }
        if curr_path_dist > best_dist_prev[&curr_id].0 {
            //info!("For curr_id {}, the curr_path_dist {} > best_dist_prev {}.", curr_id, curr_path_dist, best_dist_prev[&curr_id].0);
            // already found a shorter path to this vertex
            continue;
        }
        for neighbor_idx in edges.neighbours(&curr_id) {
            let edge_len = haversine_m_fpair(
                (*curr_pos.0, *curr_pos.1),
                nodeid_pos.get(neighbor_idx).unwrap(),
            );
            this_path_dist = curr_path_dist + OrderedFloat(edge_len);
            //info!("Looking at neighbour id {}, which is {} from start", neighbor_idx, this_path_dist);
            best_dist_prev
                .entry(*neighbor_idx)
                .and_modify(|(path_dist, prev_id)| {
                    //info!("Current best path {}", path_dist);
                    if this_path_dist < *path_dist {
                        //info!("This is a shorter path");
                        *prev_id = curr_id;
                        *path_dist = this_path_dist;
                        this_luft_dist = OrderedFloat(haversine_m(
                            curr_pos.0.0,
                            curr_pos.1.0,
                            *target_pos.0,
                            *target_pos.1,
                        ));
                        neighbor_pos_raw = nodeid_pos.get(neighbor_idx).unwrap();
                        neighbor_pos = (
                            OrderedFloat(neighbor_pos_raw.0),
                            OrderedFloat(neighbor_pos_raw.1),
                        );
                        frontier.push((
                            -(this_path_dist + this_luft_dist),
                            -this_path_dist,
                            *neighbor_idx,
                            neighbor_pos,
                        ));
                    }
                })
                .or_insert_with(|| {
                    //info!("Never seen this neighbour before, adding it to frontier");
                    this_luft_dist = OrderedFloat(haversine_m(
                        curr_pos.0.0,
                        curr_pos.1.0,
                        *target_pos.0,
                        *target_pos.1,
                    ));
                    neighbor_pos_raw = nodeid_pos.get(neighbor_idx).unwrap();
                    neighbor_pos = (
                        OrderedFloat(neighbor_pos_raw.0),
                        OrderedFloat(neighbor_pos_raw.1),
                    );
                    frontier.push((
                        -(this_path_dist + this_luft_dist),
                        -this_path_dist,
                        *neighbor_idx,
                        neighbor_pos,
                    ));
                    (this_path_dist, curr_id)
                });
        }
    }
    assert!(
        best_dist_prev.contains_key(&target_idx),
        "Something went wrong with nid {}",
        target_idx
    );

    let mut contracted_path = Vec::new();
    contracted_path.push(target_idx);
    while *contracted_path.last().unwrap() != start_idx {
        contracted_path.push(best_dist_prev[contracted_path.last().unwrap()].1);
    }
    contracted_path.reverse();

    ((start_idx, target_idx), contracted_path)
}

/// Do an A* search from the start index to the end index.
pub fn a_star_directed_single<'a>(
    start_idx: i64,
    target_idx: i64,
    nodeid_pos: &'a impl NodeIdPosition,
    inter_store: &inter_store::InterStore,
    edges: &'a DirectedGraph2,
) -> Option<f64> {
    assert!(edges.contains_vertex(&start_idx));
    assert!(edges.contains_vertex(&target_idx));
    let start_pos = nodeid_pos.get(&start_idx).unwrap();
    let target_pos = nodeid_pos.get(&target_idx).unwrap();
    let start_pos = (OrderedFloat(start_pos.0), OrderedFloat(start_pos.1));
    let target_pos = (OrderedFloat(target_pos.0), OrderedFloat(target_pos.1));

    let mut frontier = BinaryHeap::new();
    let mut best_dist_prev: HashMap<i64, (OrderedFloat<f64>, i64)> = HashMap::new();
    best_dist_prev.clear();
    best_dist_prev.insert(start_idx, (OrderedFloat(0.), start_idx));
    frontier.clear();
    frontier.push((
        OrderedFloat(-haversine_m(
            *start_pos.0,
            *start_pos.1,
            *target_pos.0,
            *target_pos.1,
        )),
        OrderedFloat(-0.),
        start_idx,
        start_pos,
    ));

    let mut this_luft_dist = Default::default();
    let mut neighbor_pos_raw = Default::default();
    let mut neighbor_pos = Default::default();
    let mut this_path_dist;

    while let Some((_est_dist, mut curr_path_dist, curr_id, curr_pos)) = frontier.pop() {
        curr_path_dist *= -1.;
        if curr_id == target_idx {
            assert!(
                best_dist_prev.contains_key(&target_idx),
                "Something went wrong with nid {}",
                target_idx
            );
            // finished
            break;
        }
        if curr_path_dist > best_dist_prev[&curr_id].0 {
            // already found a shorter path to this vertex
            continue;
        }
        for neighbor_idx in edges.out_neighbours(curr_id) {
            let edge_len = inter_store
                .inters_directed(&curr_id, &neighbor_idx)
                .map(|nid| nodeid_pos.get(&nid).unwrap())
                .tuple_windows::<(_, _)>()
                .par_bridge()
                .map(|(p1, p2)| haversine::haversine_m_fpair(p1, p2))
                .sum::<f64>();

            this_path_dist = curr_path_dist + OrderedFloat(edge_len);
            best_dist_prev
                .entry(neighbor_idx)
                .and_modify(|(path_dist, prev_id)| {
                    if this_path_dist < *path_dist {
                        // This is shorter, so update path_dist & prev_id
                        *prev_id = curr_id;
                        *path_dist = this_path_dist;
                        this_luft_dist = OrderedFloat(haversine_m(
                            curr_pos.0.0,
                            curr_pos.1.0,
                            *target_pos.0,
                            *target_pos.1,
                        ));
                        neighbor_pos_raw = nodeid_pos.get(&neighbor_idx).unwrap();
                        neighbor_pos = (
                            OrderedFloat(neighbor_pos_raw.0),
                            OrderedFloat(neighbor_pos_raw.1),
                        );
                        frontier.push((
                            -(this_path_dist + this_luft_dist),
                            -this_path_dist,
                            neighbor_idx,
                            neighbor_pos,
                        ));
                    }
                })
                .or_insert_with(|| {
                    this_luft_dist = OrderedFloat(haversine_m(
                        curr_pos.0.0,
                        curr_pos.1.0,
                        *target_pos.0,
                        *target_pos.1,
                    ));
                    neighbor_pos_raw = nodeid_pos.get(&neighbor_idx).unwrap();
                    neighbor_pos = (
                        OrderedFloat(neighbor_pos_raw.0),
                        OrderedFloat(neighbor_pos_raw.1),
                    );
                    frontier.push((
                        -(this_path_dist + this_luft_dist),
                        -this_path_dist,
                        neighbor_idx,
                        neighbor_pos,
                    ));
                    (this_path_dist, curr_id)
                });
        }
    }

    best_dist_prev
        .remove(&target_idx)
        .map(|(dist, _prev_idx)| *dist)
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
